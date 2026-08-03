"""Deterministic staged backward truncation for Qwen LoRA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

import torch


@dataclass(frozen=True)
class BackwardStage:
    """One backward regime beginning before ``start_step`` is executed."""

    start_step: int
    train_top_layers: int | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_step, bool)
            or not isinstance(self.start_step, int)
            or self.start_step < 0
        ):
            raise ValueError("backward stage start_step must be a non-negative integer")
        if self.train_top_layers is not None and (
            isinstance(self.train_top_layers, bool)
            or not isinstance(self.train_top_layers, int)
            or self.train_top_layers < 1
        ):
            raise ValueError(
                "backward stage train_top_layers must be null or a positive integer"
            )

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "start_step": self.start_step,
            "mode": "full" if self.train_top_layers is None else "suffix",
            "train_top_layers": self.train_top_layers,
        }


@dataclass(frozen=True)
class StagedBackwardPlan:
    """An ordered, monotonically narrowing sequence of backward regimes."""

    name: str
    stages: tuple[BackwardStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("staged backward plan name must be non-empty")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("staged backward plan requires at least one stage")
        if self.stages[0] != BackwardStage(
            start_step=0,
            train_top_layers=None,
        ):
            raise ValueError(
                "staged backward plan must begin with full backward at step 0"
            )

        previous_step = -1
        previous_top_layers: int | None = None
        for index, stage in enumerate(self.stages):
            if stage.start_step <= previous_step:
                raise ValueError(
                    "staged backward transition steps must be strictly increasing"
                )
            if index:
                if stage.train_top_layers is None:
                    raise ValueError(
                        "staged backward plan cannot return to full backward"
                    )
                if (
                    previous_top_layers is not None
                    and stage.train_top_layers >= previous_top_layers
                ):
                    raise ValueError(
                        "staged backward suffixes must become strictly narrower"
                    )
            previous_step = stage.start_step
            previous_top_layers = stage.train_top_layers

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stages": [stage.to_dict() for stage in self.stages],
        }


_FULL = BackwardStage(start_step=0, train_top_layers=None)

PLANS = MappingProxyType(
    {
        "full_only": StagedBackwardPlan(
            name="full_only",
            stages=(_FULL,),
        ),
        "top7_after8": StagedBackwardPlan(
            name="top7_after8",
            stages=(
                _FULL,
                BackwardStage(start_step=8, train_top_layers=7),
            ),
        ),
        "top1_after8": StagedBackwardPlan(
            name="top1_after8",
            stages=(
                _FULL,
                BackwardStage(start_step=8, train_top_layers=1),
            ),
        ),
        "top1_after10": StagedBackwardPlan(
            name="top1_after10",
            stages=(
                _FULL,
                BackwardStage(start_step=10, train_top_layers=1),
            ),
        ),
        "top1_after12": StagedBackwardPlan(
            name="top1_after12",
            stages=(
                _FULL,
                BackwardStage(start_step=12, train_top_layers=1),
            ),
        ),
        "top14_after4_top3_after8": StagedBackwardPlan(
            name="top14_after4_top3_after8",
            stages=(
                _FULL,
                BackwardStage(start_step=4, train_top_layers=14),
                BackwardStage(start_step=8, train_top_layers=3),
            ),
        ),
        "top14_after4_top1_after8": StagedBackwardPlan(
            name="top14_after4_top1_after8",
            stages=(
                _FULL,
                BackwardStage(start_step=4, train_top_layers=14),
                BackwardStage(start_step=8, train_top_layers=1),
            ),
        ),
        "top3_after4": StagedBackwardPlan(
            name="top3_after4",
            stages=(
                _FULL,
                BackwardStage(start_step=4, train_top_layers=3),
            ),
        ),
        "top1_after4": StagedBackwardPlan(
            name="top1_after4",
            stages=(
                _FULL,
                BackwardStage(start_step=4, train_top_layers=1),
            ),
        ),
    }
)


def resolve_plan(name: object) -> StagedBackwardPlan:
    if not isinstance(name, str) or name not in PLANS:
        raise RuntimeError(f"unsupported staged backward plan: {name!r}")
    return PLANS[name]


@dataclass(frozen=True)
class AppliedStage:
    """Serializable evidence for one applied runtime transition."""

    plan_name: str
    stage_index: int
    completed_updates: int
    train_top_layers: int
    boundary_index: int
    truncated_parameter_count: int
    active_parameter_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "plan_name": self.plan_name,
            "stage_index": self.stage_index,
            "completed_updates": self.completed_updates,
            "train_top_layers": self.train_top_layers,
            "boundary_index": self.boundary_index,
            "truncated_parameter_count": self.truncated_parameter_count,
            "active_parameter_count": self.active_parameter_count,
        }


@dataclass(frozen=True)
class _RuntimeStage:
    definition: BackwardStage
    boundary_index: int | None
    lower_parameters: tuple[torch.nn.Parameter, ...]
    upper_parameters: tuple[torch.nn.Parameter, ...]


class StagedBackwardTruncation:
    """Apply every predeclared detach boundary at its exact optimizer-step index.

    LoRA parameters below the active boundary remain unchanged in the forward path
    and remain part of the standard saved PEFT adapter. The boundary only prevents
    later backward graphs from traversing the frozen prefix.
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        plan: StagedBackwardPlan,
        *,
        retire_prefix_parameters: bool = False,
    ) -> None:
        self.layers = tuple(layers)
        self.plan = plan
        self.retire_prefix_parameters = retire_prefix_parameters
        self._hook = None
        self._active_stage_index = 0
        self._transitions: list[AppliedStage] = []
        self._retired_parameters: list[torch.nn.Parameter] = []
        if not self.layers:
            raise RuntimeError("staged backward truncation requires decoder layers")

        all_parameters = tuple(
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
        if not all_parameters:
            raise RuntimeError(
                "staged backward truncation requires trainable parameters"
            )
        self._all_parameters = all_parameters

        runtimes = []
        for stage in plan.stages:
            if stage.train_top_layers is None:
                runtime = _RuntimeStage(
                    definition=stage,
                    boundary_index=None,
                    lower_parameters=(),
                    upper_parameters=all_parameters,
                )
            else:
                if stage.train_top_layers >= len(self.layers):
                    raise RuntimeError(
                        "staged trainable suffix must be smaller than the model"
                    )
                boundary_index = len(self.layers) - stage.train_top_layers - 1
                lower = tuple(
                    parameter
                    for layer in self.layers[: boundary_index + 1]
                    for parameter in layer.parameters()
                    if parameter.requires_grad
                )
                upper = tuple(
                    parameter
                    for layer in self.layers[boundary_index + 1 :]
                    for parameter in layer.parameters()
                    if parameter.requires_grad
                )
                if not lower or not upper:
                    raise RuntimeError(
                        "staged boundary must have trainable parameters on both sides"
                    )
                runtime = _RuntimeStage(
                    definition=stage,
                    boundary_index=boundary_index,
                    lower_parameters=lower,
                    upper_parameters=upper,
                )
            runtimes.append(runtime)
        self._runtime_stages = tuple(runtimes)

    @staticmethod
    def _detach_hidden_state(_module, _inputs, output):
        if torch.is_tensor(output):
            return output.detach()
        if (
            isinstance(output, tuple)
            and len(output) == 1
            and torch.is_tensor(output[0])
        ):
            return (output[0].detach(),)
        raise RuntimeError(
            "staged decoder boundary returned an unsupported output"
        )

    @property
    def active(self) -> bool:
        return self._active_stage_index > 0

    @property
    def hook_active(self) -> bool:
        return self._hook is not None

    @property
    def stage_index(self) -> int:
        return self._active_stage_index

    @property
    def current_stage(self) -> BackwardStage:
        return self._runtime_stages[self._active_stage_index].definition

    @property
    def boundary_index(self) -> int | None:
        return self._runtime_stages[self._active_stage_index].boundary_index

    @property
    def lower_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self._runtime_stages[self._active_stage_index].lower_parameters

    @property
    def upper_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self._runtime_stages[self._active_stage_index].upper_parameters

    @property
    def truncated_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.lower_parameters)

    @property
    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.upper_parameters)

    @property
    def transition_history(self) -> tuple[AppliedStage, ...]:
        return tuple(self._transitions)

    @property
    def retired_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self._retired_parameters)

    def validate_trainable_partition(
        self,
        trainable_parameters: Sequence[torch.nn.Parameter],
    ) -> None:
        expected = {id(parameter) for parameter in trainable_parameters}
        if expected != {id(parameter) for parameter in self._all_parameters}:
            raise RuntimeError(
                "staged decoder layers do not contain the complete trainable adapter"
            )
        for runtime in self._runtime_stages:
            lower = {id(parameter) for parameter in runtime.lower_parameters}
            upper = {id(parameter) for parameter in runtime.upper_parameters}
            if lower & upper or lower | upper != expected:
                raise RuntimeError(
                    "staged lower/upper parameters do not exactly partition "
                    "the trainable adapter"
                )

    def validate_total_steps(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise RuntimeError("staged backward total steps must be a positive integer")
        if (
            len(self.plan.stages) > 1
            and steps <= self.plan.stages[-1].start_step
        ):
            raise RuntimeError(
                "staged backward plan requires at least one update in its final stage"
            )

    def transition_before_step(self, step: int) -> AppliedStage | None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RuntimeError("staged backward step must be a non-negative integer")

        target_index = max(
            index
            for index, runtime in enumerate(self._runtime_stages)
            if runtime.definition.start_step <= step
        )
        if target_index == self._active_stage_index:
            return None
        if target_index != self._active_stage_index + 1:
            raise RuntimeError("staged backward transition was skipped")

        runtime = self._runtime_stages[target_index]
        if step != runtime.definition.start_step:
            raise RuntimeError("staged backward transition was skipped")
        if any(parameter.grad is not None for parameter in self._all_parameters):
            raise RuntimeError(
                "adapter gradients were not cleared before staged transition"
            )
        if (
            runtime.boundary_index is None
            or runtime.definition.train_top_layers is None
        ):
            raise RuntimeError(
                "staged backward transition cannot restore full backward"
            )

        if self.retire_prefix_parameters:
            already_retired = {id(parameter) for parameter in self._retired_parameters}
            newly_retired = tuple(
                parameter
                for parameter in runtime.lower_parameters
                if id(parameter) not in already_retired
            )
            for parameter in newly_retired:
                if not parameter.requires_grad:
                    raise RuntimeError(
                        "new staged-prefix parameter was already non-trainable"
                    )
                parameter.requires_grad_(False)
            self._retired_parameters.extend(newly_retired)
        else:
            new_hook = self.layers[runtime.boundary_index].register_forward_hook(
                self._detach_hidden_state
            )
            old_hook = self._hook
            self._hook = new_hook
            if old_hook is not None:
                old_hook.remove()
        self._active_stage_index = target_index

        applied = AppliedStage(
            plan_name=self.plan.name,
            stage_index=target_index,
            completed_updates=step,
            train_top_layers=runtime.definition.train_top_layers,
            boundary_index=runtime.boundary_index,
            truncated_parameter_count=sum(
                parameter.numel() for parameter in runtime.lower_parameters
            ),
            active_parameter_count=sum(
                parameter.numel() for parameter in runtime.upper_parameters
            ),
        )
        self._transitions.append(applied)
        return applied

    def assert_gradient_partition(self) -> None:
        if not self.active:
            return
        if any(parameter.grad is not None for parameter in self.lower_parameters):
            raise RuntimeError(
                "staged-truncated parameters unexpectedly received gradients"
            )
        if not all(parameter.grad is not None for parameter in self.upper_parameters):
            raise RuntimeError(
                "not every active staged-suffix parameter received a gradient"
            )

    def assert_completed(self) -> None:
        if self._active_stage_index != len(self._runtime_stages) - 1:
            raise RuntimeError("staged backward plan did not apply every transition")
        expected_steps = tuple(
            stage.start_step for stage in self.plan.stages[1:]
        )
        observed_steps = tuple(
            transition.completed_updates for transition in self._transitions
        )
        if observed_steps != expected_steps:
            raise RuntimeError("staged backward transition history differs from plan")

    def summary(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "implementation": (
                "retired-prefix-requires-grad"
                if self.retire_prefix_parameters
                else "detach-boundary-hook"
            ),
            "applied_transitions": [
                transition.to_dict() for transition in self._transitions
            ],
        }

    def restore_for_export(self) -> None:
        if any(parameter.grad is not None for parameter in self._retired_parameters):
            raise RuntimeError(
                "retired prefix gradients must be clear before export restoration"
            )
        for parameter in self._retired_parameters:
            parameter.requires_grad_(True)
        self._retired_parameters.clear()

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
