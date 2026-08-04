"""Minimal, static LoRA injection for the pinned Qwen2.5 architecture.

The runtime surface deliberately mirrors the small subset of PEFT's ``Linear``
wrapper consumed by this candidate's reviewed projection implementations.  It
does not provide a generic adapter API: unsupported model shapes and adapter
features fail closed during injection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


ADAPTER_NAME = "default"
EXPECTED_QWEN2_LAYERS = 28
PROJECTION_PATHS = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
)


def _new_lora_pair(
    base_layer: torch.nn.Linear,
    *,
    rank: int,
) -> tuple[torch.nn.Linear, torch.nn.Linear]:
    """Construct one pair in PEFT 0.19.1's exact RNG and cast order."""

    lora_a = torch.nn.Linear(base_layer.in_features, rank, bias=False)
    lora_b = torch.nn.Linear(rank, base_layer.out_features, bias=False)
    torch.nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(lora_b.weight)
    lora_a.to(
        device=base_layer.weight.device,
        dtype=base_layer.weight.dtype,
    )
    lora_b.to(
        device=base_layer.weight.device,
        dtype=base_layer.weight.dtype,
    )
    return lora_a, lora_b


class StaticLoraLinear(torch.nn.Module):
    """A single enabled, unmerged, zero-dropout LoRA over ``nn.Linear``."""

    def __init__(
        self,
        base_layer: torch.nn.Linear,
        *,
        rank: int,
        alpha: int,
    ) -> None:
        super().__init__()
        if type(base_layer) is not torch.nn.Linear:
            raise RuntimeError("static LoRA supports only exact torch.nn.Linear")
        if type(rank) is not int or rank < 1:
            raise RuntimeError("static LoRA rank must be a positive integer")
        if type(alpha) is not int or alpha < 1:
            raise RuntimeError("static LoRA alpha must be a positive integer")
        if base_layer.weight.ndim != 2:
            raise RuntimeError("static LoRA base weight must be two-dimensional")
        if base_layer.weight.requires_grad or (
            base_layer.bias is not None and base_layer.bias.requires_grad
        ):
            raise RuntimeError("static LoRA base projection must already be frozen")

        self.base_layer = base_layer

        # Match PEFT 0.19.1's initialization order and RNG consumption: construct
        # A then B in the default CPU dtype, reset A a second time, zero B, and
        # only then move the adapter to its base projection.
        lora_a, lora_b = _new_lora_pair(
            base_layer,
            rank=rank,
        )

        self.lora_A = torch.nn.ModuleDict({ADAPTER_NAME: lora_a})
        self.lora_B = torch.nn.ModuleDict({ADAPTER_NAME: lora_b})
        self.lora_dropout = torch.nn.ModuleDict(
            {ADAPTER_NAME: torch.nn.Identity()}
        )
        self.scaling = {ADAPTER_NAME: alpha / rank}
        self.r = {ADAPTER_NAME: rank}
        self.lora_alpha = {ADAPTER_NAME: alpha}
        self.lora_variant = {}
        self.disable_adapters = False
        self.merged = False
        self.cast_input_dtype_enabled = True
        self.active_adapters = (ADAPTER_NAME,)

    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.cast_input_dtype_enabled:
            return x.to(dtype=dtype)
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if args or kwargs:
            raise RuntimeError(
                "static LoRA projection accepts only its input tensor"
            )
        if self.disable_adapters or self.merged:
            raise RuntimeError("static LoRA adapter state changed unexpectedly")
        if self.active_adapters != (ADAPTER_NAME,):
            raise RuntimeError("static LoRA active adapter changed unexpectedly")

        result = self.base_layer(x)
        result_dtype = result.dtype
        lora_a = self.lora_A[ADAPTER_NAME]
        lora_b = self.lora_B[ADAPTER_NAME]
        adapter_input = self._cast_input_dtype(x, lora_a.weight.dtype)
        adapter_output = lora_b(
            lora_a(self.lora_dropout[ADAPTER_NAME](adapter_input))
        )
        return (
            result + adapter_output * self.scaling[ADAPTER_NAME]
        ).to(result_dtype)


@dataclass(frozen=True)
class StaticLoraInjection:
    """Authenticated inventory returned by the direct injector."""

    model: torch.nn.Module
    transformer: torch.nn.Module
    wrappers: tuple[StaticLoraLinear, ...]

    @property
    def trainable_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(
            parameter
            for wrapper in self.wrappers
            for parameter in (
                wrapper.lora_A[ADAPTER_NAME].weight,
                wrapper.lora_B[ADAPTER_NAME].weight,
            )
        )


@dataclass(frozen=True)
class DirectParentDispatchSpec:
    """Packed adapter tensors consumed directly by one parent dispatch."""

    kind: str
    base_layers: tuple[torch.nn.Linear, ...]
    packed_lora_a: torch.nn.Parameter
    lora_b_weights: tuple[torch.nn.Parameter, ...]
    scalings: tuple[float, ...]
    output_base: torch.nn.Linear | None = None
    output_lora_a: torch.nn.Parameter | None = None
    output_lora_b: torch.nn.Parameter | None = None
    output_scaling: float | None = None


@dataclass(frozen=True)
class DirectPackedLoraLayer:
    """The two direct packed parent specifications for one decoder layer."""

    attention: DirectParentDispatchSpec
    mlp: DirectParentDispatchSpec


@dataclass(frozen=True)
class DirectPackedLoraInjection:
    """Wrapper-free packed runtime and canonical export inventory."""

    model: torch.nn.Module
    transformer: torch.nn.Module
    layers: tuple[DirectPackedLoraLayer, ...]
    rank: int
    alpha: int

    @property
    def trainable_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        parameters = []
        for layer in self.layers:
            attention = layer.attention
            mlp = layer.mlp
            parameters.extend(
                (
                    attention.packed_lora_a,
                    *attention.lora_b_weights,
                    attention.output_lora_a,
                    attention.output_lora_b,
                    mlp.packed_lora_a,
                    *mlp.lora_b_weights,
                )
            )
        if any(parameter is None for parameter in parameters):
            raise RuntimeError("direct packed adapter inventory is incomplete")
        return tuple(parameters)

    def parent_dispatch_spec(
        self,
        layer_index: int,
        kind: str,
    ) -> DirectParentDispatchSpec:
        if (
            type(layer_index) is not int
            or not 0 <= layer_index < len(self.layers)
        ):
            raise RuntimeError("direct packed layer index is invalid")
        layer = self.layers[layer_index]
        if kind == "attention":
            return layer.attention
        if kind == "mlp":
            return layer.mlp
        raise RuntimeError(f"unknown direct packed parent kind: {kind!r}")


def _resolve_projection(layer: torch.nn.Module, path: tuple[str, str]):
    owner_name, projection_name = path
    try:
        owner = getattr(layer, owner_name)
        projection = getattr(owner, projection_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"pinned Qwen2 projection is unavailable: {owner_name}.{projection_name}"
        ) from exc
    return owner, projection_name, projection


def inject_static_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
) -> StaticLoraInjection:
    """Inject the exact six LoRA targets in each of 28 pinned decoder layers."""

    expected_targets = tuple(name for _owner, name in PROJECTION_PATHS)
    if tuple(target_modules) != expected_targets:
        raise RuntimeError(
            "static LoRA target order must exactly match the pinned six projections"
        )
    try:
        transformer = model.model
        layers = tuple(transformer.layers)
    except AttributeError as exc:
        raise RuntimeError("pinned Qwen2 decoder structure is unavailable") from exc
    if len(layers) != EXPECTED_QWEN2_LAYERS:
        raise RuntimeError(
            "static LoRA requires exactly "
            f"{EXPECTED_QWEN2_LAYERS} decoder layers; got {len(layers)}"
        )

    # Freeze before constructing any adapter so only adapter parameters enter the
    # optimizer, just as with PEFT's default LoRA injection.
    model.requires_grad_(False)

    resolved = []
    seen = set()
    for layer_index, layer in enumerate(layers):
        for path in PROJECTION_PATHS:
            owner, projection_name, projection = _resolve_projection(layer, path)
            if type(projection) is not torch.nn.Linear:
                raise RuntimeError(
                    f"layer {layer_index} {'.'.join(path)} is not exact nn.Linear"
                )
            if id(projection) in seen:
                raise RuntimeError("a base projection appears more than once")
            seen.add(id(projection))
            resolved.append((owner, projection_name, projection))

    installed: list[tuple[torch.nn.Module, str, torch.nn.Linear]] = []
    wrappers: list[StaticLoraLinear] = []
    try:
        for owner, projection_name, projection in resolved:
            wrapper = StaticLoraLinear(
                projection,
                rank=rank,
                alpha=alpha,
            )
            setattr(owner, projection_name, wrapper)
            installed.append((owner, projection_name, projection))
            wrappers.append(wrapper)
    except Exception:
        for owner, projection_name, projection in reversed(installed):
            setattr(owner, projection_name, projection)
        raise

    injection = StaticLoraInjection(
        model=model,
        transformer=transformer,
        wrappers=tuple(wrappers),
    )
    observed = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    expected = injection.trainable_parameters
    if tuple(map(id, observed)) != tuple(map(id, expected)):
        # Preserve a valid frozen base even if a future model changes traversal.
        raise RuntimeError(
            "static LoRA trainable traversal differs from the injected adapter"
        )
    return injection


def _take_weight(module: torch.nn.Linear) -> torch.nn.Parameter:
    weight = module.weight
    delattr(module, "weight")
    return weight


def _register_direct_parameter(
    owner: torch.nn.Module,
    name: str,
    parameter: torch.nn.Parameter,
    installed: list[tuple[torch.nn.Module, str]],
) -> None:
    if hasattr(owner, name):
        raise RuntimeError(f"direct packed parameter already exists: {name}")
    owner.register_parameter(name, parameter)
    installed.append((owner, name))


def inject_direct_packed_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
) -> DirectPackedLoraInjection:
    """Construct the combined runtime directly, without temporary wrappers.

    Per-projection A/B ``Linear`` objects exist only long enough to reproduce
    PEFT's constructor/reset/cast RNG sequence. Their parameters are immediately
    registered in the packed parent representation; no static or PEFT LoRA
    wrapper is ever installed.
    """

    expected_targets = tuple(name for _owner, name in PROJECTION_PATHS)
    if tuple(target_modules) != expected_targets:
        raise RuntimeError(
            "direct packed LoRA target order must match the pinned projections"
        )
    if type(rank) is not int or rank < 1:
        raise RuntimeError("direct packed LoRA rank must be a positive integer")
    if type(alpha) is not int or alpha < 1:
        raise RuntimeError("direct packed LoRA alpha must be a positive integer")
    try:
        transformer = model.model
        layers = tuple(transformer.layers)
    except AttributeError as exc:
        raise RuntimeError("pinned Qwen2 decoder structure is unavailable") from exc
    if len(layers) != EXPECTED_QWEN2_LAYERS:
        raise RuntimeError(
            "direct packed LoRA requires exactly "
            f"{EXPECTED_QWEN2_LAYERS} decoder layers; got {len(layers)}"
        )

    resolved_layers = []
    seen = set()
    for layer_index, layer in enumerate(layers):
        resolved = []
        for path in PROJECTION_PATHS:
            owner, projection_name, projection = _resolve_projection(layer, path)
            if type(projection) is not torch.nn.Linear:
                raise RuntimeError(
                    f"layer {layer_index} {'.'.join(path)} is not exact nn.Linear"
                )
            if id(projection) in seen:
                raise RuntimeError("a direct packed base projection is shared")
            seen.add(id(projection))
            resolved.append((owner, projection_name, projection))
        resolved_layers.append(tuple(resolved))

    model.requires_grad_(False)
    scaling = alpha / rank
    installed: list[tuple[torch.nn.Module, str]] = []
    direct_layers = []
    try:
        for layer_index, (layer, resolved) in enumerate(
            zip(layers, resolved_layers)
        ):
            # This loop order is q, k, v, o, gate, up: the exact module traversal
            # order used by PEFT injection.
            pairs = tuple(
                _new_lora_pair(projection, rank=rank)
                for _owner, _name, projection in resolved
            )
            a_weights = tuple(_take_weight(pair[0]) for pair in pairs)
            b_weights = tuple(_take_weight(pair[1]) for pair in pairs)
            q, k, v, o, gate, up = (
                projection for _owner, _name, projection in resolved
            )
            q_a, k_a, v_a, o_a, gate_a, up_a = a_weights
            q_b, k_b, v_b, o_b, gate_b, up_b = b_weights

            qkv_a = torch.nn.Parameter(
                torch.cat(
                    [q_a.detach(), k_a.detach(), v_a.detach()],
                    dim=0,
                ),
                requires_grad=True,
            )
            gate_up_a = torch.nn.Parameter(
                torch.cat([gate_a.detach(), up_a.detach()], dim=0),
                requires_grad=True,
            )
            attention = layer.self_attn
            mlp = layer.mlp
            names_and_parameters = (
                (attention, "_direct_qkv_lora_a_weight", qkv_a),
                (attention, "_direct_q_lora_b_weight", q_b),
                (attention, "_direct_k_lora_b_weight", k_b),
                (attention, "_direct_v_lora_b_weight", v_b),
                (attention, "_direct_o_lora_a_weight", o_a),
                (attention, "_direct_o_lora_b_weight", o_b),
                (mlp, "_direct_gate_up_lora_a_weight", gate_up_a),
                (mlp, "_direct_gate_lora_b_weight", gate_b),
                (mlp, "_direct_up_lora_b_weight", up_b),
            )
            for owner, name, parameter in names_and_parameters:
                _register_direct_parameter(owner, name, parameter, installed)

            attention_spec = DirectParentDispatchSpec(
                kind="attention",
                base_layers=(q, k, v),
                packed_lora_a=qkv_a,
                lora_b_weights=(q_b, k_b, v_b),
                scalings=(scaling, scaling, scaling),
                output_base=o,
                output_lora_a=o_a,
                output_lora_b=o_b,
                output_scaling=scaling,
            )
            mlp_spec = DirectParentDispatchSpec(
                kind="mlp",
                base_layers=(gate, up),
                packed_lora_a=gate_up_a,
                lora_b_weights=(gate_b, up_b),
                scalings=(scaling, scaling),
            )
            direct_layers.append(
                DirectPackedLoraLayer(
                    attention=attention_spec,
                    mlp=mlp_spec,
                )
            )
    except Exception:
        for owner, name in reversed(installed):
            delattr(owner, name)
        raise

    injection = DirectPackedLoraInjection(
        model=model,
        transformer=transformer,
        layers=tuple(direct_layers),
        rank=rank,
        alpha=alpha,
    )
    observed = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    expected = injection.trainable_parameters
    if {id(parameter) for parameter in observed} != {
        id(parameter) for parameter in expected
    } or len(observed) != len(expected):
        for owner, name in reversed(installed):
            delattr(owner, name)
        raise RuntimeError(
            "direct packed trainable traversal differs from its adapter inventory"
        )
    return injection
