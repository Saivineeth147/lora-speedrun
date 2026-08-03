"""Track 1: pinned-Qwen loading, prefix batching, and a short fixed tail.

Core runtime:
  1. A static 0.50-epoch request with 8e-4 peak LR and four warmup updates.
  2. GSM8K `<<...>>` calculator annotations stripped (pure formatting) — ~13% fewer tokens.
  3. No HF Trainer: a hand-rolled loop over pre-packed fixed-length GPU-resident blocks
     (zero dataloader/collate overhead, no attention mask needed — all blocks are full).
  4. Chunked completion-only cross-entropy as a custom autograd.Function: lm_head logits
     are never materialized for the whole batch, and are only computed for the ~70% of
     tokens that carry labels. Saves ~10 GB of fp32 logits traffic per step.
  5. The exact pinned safetensors file can be warmed in a background A/B while
     Torch imports; the direct loader bypasses Auto*, from_pretrained, Hub, and
     Accelerate, and the Rust tokenizer reads tokenizer.json directly.
  6. LoRA parameters stay in the base model's BF16 dtype instead of PEFT's default FP32
     autocast, avoiding full-size FP32 adapter-input saves and per-layer dtype casts.
  7. Deterministic best-fit repacks exactly the examples emitted by the baseline's
     next-fit loop, never including its final unflushed buffer.
  8. Frozen q/k/v base projections and their shared-input LoRA A projections each
     execute as one packed GEMM per layer during training.
 9. Frozen gate/up base projections and their shared-input LoRA A projections each
     execute as one packed GEMM per layer during training.
 10. The seeded epoch permutation, completion indices, and completion targets are
     materialized once; training batches are contiguous views with one final loss sync.
 11. Eight full backward updates are followed by four updates through only the
     top decoder layer, while every lower-layer LoRA delta remains active.
 12. The frozen layer-26 boundary can be materialized in larger no-gradient batches
     and consumed directly by layer 27 plus the final RMSNorm.
 13. The four tail updates run at the bounded 8e-4 peak while updates one through
     eight remain exactly unchanged from the reference schedule.

Contract: python train.py --data-dir <gsm8k_train> --output-dir <dir> --seed <int>
"""

import argparse
import glob
import json
import os
import threading
import time

T0 = time.monotonic()


def log(msg):
    print(f"[t+{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


def emit_phase(name, *, started=None, **fields):
    now = time.monotonic()
    payload = {
        "name": name,
        "timestamp_seconds": now - T0,
        **fields,
    }
    if started is not None:
        payload["duration_seconds"] = now - started
    log(
        "phase="
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return payload["timestamp_seconds"]


BASE_MODEL = "Qwen/Qwen2.5-1.5B"
BASE_MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
EXPECTED_MODEL_SAFETENSORS_BYTES = 3_087_467_144
_STARTUP_PAGE_WARMER = "on"
_WARMER_DONE = threading.Event()
_WARMER_OUTCOME = {
    "status": "pending",
    "success": False,
    "expected_bytes": 0,
    "read_bytes": 0,
    "completed_at_seconds": None,
    "error": None,
}


def _warm_model_files():
    """Read only the exact pinned model file while Python imports Torch."""
    started = time.monotonic()
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    path = os.path.join(
        hf,
        "hub",
        "models--Qwen--Qwen2.5-1.5B",
        "snapshots",
        BASE_MODEL_REVISION,
        "model.safetensors",
    )
    read_bytes = 0
    status = "error"
    error = None
    try:
        expected_bytes = os.path.getsize(path)
        if expected_bytes != EXPECTED_MODEL_SAFETENSORS_BYTES:
            raise RuntimeError("pinned model file size differs")
        with open(path, "rb") as stream:
            while chunk := stream.read(1 << 25):
                read_bytes += len(chunk)
        if read_bytes != expected_bytes:
            raise RuntimeError("pinned model file warmer read was incomplete")
        status = "ok"
    except (OSError, RuntimeError) as exc:
        error = exc
    finally:
        completed_at_seconds = emit_phase(
            "page_warming",
            started=started,
            expected_bytes=EXPECTED_MODEL_SAFETENSORS_BYTES,
            read_bytes=read_bytes,
            mode="on",
            status=status,
            success=status == "ok",
        )
        _WARMER_OUTCOME.update(
            {
                "status": status,
                "success": status == "ok",
                "expected_bytes": EXPECTED_MODEL_SAFETENSORS_BYTES,
                "read_bytes": read_bytes,
                "completed_at_seconds": completed_at_seconds,
                "error": error,
            }
        )
        _WARMER_DONE.set()


if _STARTUP_PAGE_WARMER == "on":
    _WARMER_THREAD = threading.Thread(
        target=_warm_model_files,
        name="pinned-model-page-warmer",
        daemon=True,
    )
    _WARMER_THREAD.start()
else:
    _WARMER_THREAD = None
    _WARMER_COMPLETED_AT_SECONDS = emit_phase(
        "page_warming",
        started=T0,
        expected_bytes=0,
        read_bytes=0,
        mode="off",
        status="disabled",
        success=False,
    )
    _WARMER_OUTCOME.update(
        {
            "status": "disabled",
            "success": False,
            "expected_bytes": 0,
            "read_bytes": 0,
            "completed_at_seconds": _WARMER_COMPLETED_AT_SECONDS,
            "error": None,
        }
    )
    _WARMER_DONE.set()

import random  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from types import MethodType  # noqa: E402

_TORCH_IMPORT_STARTED = time.monotonic()
import torch  # noqa: E402
emit_phase("torch_import", started=_TORCH_IMPORT_STARTED)

import numpy as np  # noqa: E402

from area_schedule import (  # noqa: E402
    learning_rate_at,
    learning_rate_sequence,
    schedule_config,
)
from engineering_config import (  # noqa: E402
    ENGINEERING_FEATURES,
    ENGINEERING_VARIANT,
)
from hot_loop_plan import build_epoch_plan  # noqa: E402
from packing import best_fit_pack_baseline_membership  # noqa: E402
from production_config import (  # noqa: E402
    ADAM_BETA2 as ADAM_B2,
    BACKWARD_PLAN_NAME,
    BATCH_SIZE as BS,
    CE_CHUNK_SIZE as CE_CHUNK,
    EPOCH_FRACTION as EPOCHS,
    EXPECTED_SAVED_ADAPTER_PARAMETERS,
    FULL_BACKWARD_UPDATES,
    LEARNING_RATE as LR,
    LORA_ALPHA as ALPHA,
    LORA_RANK as RANK,
    MAX_OPTIMIZER_STEPS,
    PACK_LENGTH as PACK_LEN,
    PAGE_WARMER,
    PREFIX_CACHE_BATCH,
    PRODUCTION_FEATURES,
    STARTUP_LOADER,
    STRIP_CALCULATOR_ANNOTATIONS as STRIP_ANNOT,
    TAIL_SCHEDULE_NAME,
    TARGET_MODULES,
    TRAIN_SUBSET as SUBSET,
    WARMUP_STEPS as WARMUP,
    projection_components_for,
)
from projection_ablations import (  # noqa: E402
    fuse_qwen2_lora_a_gate_up,
    fuse_qwen2_lora_a_qkv,
    pack_qwen2_base_gate_up,
    pack_qwen2_base_qkv,
)
from progressive import (  # noqa: E402
    StagedBackwardTruncation,
    resolve_plan,
)
from step_frontier import (  # noqa: E402
    build_shuffled_block_order,
    build_training_horizon,
)


STAGED_PLAN = resolve_plan(BACKWARD_PLAN_NAME)

_PACKED_QKV_WEIGHT = "_packed_base_qkv_weight"
_PACKED_QKV_BIAS = "_packed_base_qkv_bias"
_PACKED_GATE_UP_WEIGHT = "_packed_base_gate_up_weight"
_PACKED_GATE_UP_BIAS = "_packed_base_gate_up_bias"
_FUSED_A_PARAMETER = "_packed_a_weight"


def _validate_packed_adapter_requires_grad(
    *,
    label,
    packed_a,
    b_weights,
):
    """Require one coherent trainable or intentionally retired adapter group."""

    packed_a_requires_grad = packed_a.requires_grad
    if any(
        b_weight.requires_grad != packed_a_requires_grad
        for b_weight in b_weights
    ):
        raise RuntimeError(
            f"{label}: packed LoRA A/B requires_grad states diverged"
        )


class _PackedQKVAndLoraAGroup:
    """Pack frozen base and LoRA-A q/k/v projections under one wrapper owner."""

    def __init__(self, modules, label):
        self.modules = tuple(modules)
        self.label = label
        self.adapter = tuple(self.modules[0].active_adapters)[0]
        self.base_layers = tuple(module.base_layer for module in self.modules)
        self.a_layers = tuple(module.lora_A[self.adapter] for module in self.modules)
        self.b_layers = tuple(module.lora_B[self.adapter] for module in self.modules)
        self.b_weights = tuple(layer.weight for layer in self.b_layers)
        self.output_widths = tuple(base.out_features for base in self.base_layers)
        self.ranks = tuple(layer.weight.shape[0] for layer in self.a_layers)
        self.bias_present = tuple(base.bias is not None for base in self.base_layers)
        self.b_shapes = tuple(layer.weight.shape for layer in self.b_layers)
        self.cast_input_dtype_enabled = tuple(
            module.cast_input_dtype_enabled for module in self.modules
        )
        self.dtype = self.base_layers[0].weight.dtype
        self.adapter_dtype = self.a_layers[0].weight.dtype
        self.device = self.base_layers[0].weight.device
        self._pending_input = None
        self._pending_base_pieces = None
        self._pending_a_pieces = None
        self._next_module = 0
        self._materialized = False

        packed_weight = torch.cat(
            [base.weight.detach() for base in self.base_layers],
            dim=0,
        )
        packed_bias = torch.cat(
            [
                base.bias.detach()
                if base.bias is not None
                else base.weight.new_zeros(base.out_features)
                for base in self.base_layers
            ],
            dim=0,
        )
        packed_a = torch.nn.Parameter(
            torch.cat([layer.weight.detach() for layer in self.a_layers], dim=0),
            requires_grad=True,
        )
        self.packed_a = packed_a
        original_base_weights = tuple(base.weight for base in self.base_layers)
        original_base_biases = tuple(base.bias for base in self.base_layers)
        original_a_weights = tuple(layer.weight for layer in self.a_layers)

        host = self.modules[0]
        state_hook = None
        try:
            host.register_buffer(
                _PACKED_QKV_WEIGHT,
                packed_weight,
                persistent=False,
            )
            host.register_buffer(
                _PACKED_QKV_BIAS,
                packed_bias,
                persistent=False,
            )
            host.register_parameter(_FUSED_A_PARAMETER, packed_a)
            for base in self.base_layers:
                delattr(base, "weight")
                delattr(base, "bias")
            for layer in self.a_layers:
                delattr(layer, "weight")
            state_hook = host.register_state_dict_pre_hook(
                self._reject_packed_state_dict
            )
            for index, module in enumerate(self.modules):
                module.forward = MethodType(self._make_forward(index), module)
        except Exception:
            for module in self.modules:
                if "forward" in module.__dict__:
                    delattr(module, "forward")
            if state_hook is not None:
                state_hook.remove()
            if _FUSED_A_PARAMETER in host._parameters:
                delattr(host, _FUSED_A_PARAMETER)
            if _PACKED_QKV_BIAS in host._buffers:
                delattr(host, _PACKED_QKV_BIAS)
            if _PACKED_QKV_WEIGHT in host._buffers:
                delattr(host, _PACKED_QKV_WEIGHT)
            for base, weight, bias in zip(
                self.base_layers,
                original_base_weights,
                original_base_biases,
            ):
                if "weight" not in base._parameters:
                    base.register_parameter("weight", weight)
                if "bias" not in base._parameters:
                    base.register_parameter("bias", bias)
            for layer, weight in zip(self.a_layers, original_a_weights):
                if "weight" not in layer._parameters:
                    layer.register_parameter("weight", weight)
            raise
        self._state_hook = state_hook

    @staticmethod
    def validate(modules, label):
        modules = tuple(modules)
        if len(modules) != 3:
            raise RuntimeError(f"{label}: exactly q, k, and v modules are required")
        if len({id(module) for module in modules}) != 3:
            raise RuntimeError(f"{label}: q, k, and v modules must be distinct")

        active = tuple(modules[0].active_adapters)
        if len(active) != 1:
            raise RuntimeError(f"{label}: exactly one active adapter is required")
        adapter = active[0]
        first_base = modules[0].base_layer
        if type(first_base) is not torch.nn.Linear:
            raise RuntimeError(
                f"{label}: only torch.nn.Linear base layers are supported"
            )
        dtype = first_base.weight.dtype
        device = first_base.weight.device
        in_features = first_base.in_features
        try:
            adapter_dtype = modules[0].lora_A[adapter].weight.dtype
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(f"{label}: active LoRA A weight is unavailable") from exc
        cast_input_dtype_enabled = modules[0].cast_input_dtype_enabled
        if type(cast_input_dtype_enabled) is not bool:
            raise RuntimeError(f"{label}: PEFT input-cast state must be boolean")

        for module in modules:
            if "forward" in module.__dict__:
                raise RuntimeError(
                    f"{label}: module already has an instance forward override"
                )
            if (
                hasattr(module, _PACKED_QKV_WEIGHT)
                or hasattr(module, _PACKED_QKV_BIAS)
                or hasattr(module, _FUSED_A_PARAMETER)
            ):
                raise RuntimeError(f"{label}: module is already projection-packed")
            if module.disable_adapters or module.merged:
                raise RuntimeError(f"{label}: adapters must be enabled and unmerged")
            if tuple(module.active_adapters) != (adapter,):
                raise RuntimeError(f"{label}: active adapters do not match")
            if (
                tuple(module.lora_A.keys()) != (adapter,)
                or tuple(module.lora_B.keys()) != (adapter,)
                or tuple(module.lora_dropout.keys()) != (adapter,)
                or tuple(module.scaling.keys()) != (adapter,)
            ):
                raise RuntimeError(
                    f"{label}: exactly one vanilla LoRA adapter key is required"
                )
            if module.lora_variant:
                raise RuntimeError(f"{label}: LoRA variants are not supported")
            if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
                raise RuntimeError(f"{label}: LoRA dropout must be zero")
            if module.cast_input_dtype_enabled != cast_input_dtype_enabled:
                raise RuntimeError(f"{label}: PEFT input-cast states do not match")

            base = module.base_layer
            if type(base) is not torch.nn.Linear:
                raise RuntimeError(
                    f"{label}: only torch.nn.Linear base layers are supported"
                )
            weight = base.weight
            if (
                weight.ndim != 2
                or base.in_features != in_features
                or weight.shape != (base.out_features, in_features)
            ):
                raise RuntimeError(
                    f"{label}: base projection dimensions do not match"
                )
            if weight.dtype != dtype or weight.device != device:
                raise RuntimeError(
                    f"{label}: base projection dtype/device do not match"
                )
            if weight.requires_grad:
                raise RuntimeError(
                    f"{label}: base projection weights must be frozen"
                )
            if base.bias is not None:
                if (
                    base.bias.shape != (base.out_features,)
                    or base.bias.dtype != dtype
                    or base.bias.device != device
                ):
                    raise RuntimeError(
                        f"{label}: base bias shape/dtype/device does not match"
                    )
                if base.bias.requires_grad:
                    raise RuntimeError(
                        f"{label}: base projection biases must be frozen"
                    )

            a_layer = module.lora_A[adapter]
            b_layer = module.lora_B[adapter]
            a_weight = a_layer.weight
            b_weight = b_layer.weight
            if a_layer.bias is not None or b_layer.bias is not None:
                raise RuntimeError(f"{label}: LoRA A/B bias is not supported")
            if (
                a_weight.ndim != 2
                or a_weight.shape[1] != in_features
                or b_weight.shape != (base.out_features, a_weight.shape[0])
            ):
                raise RuntimeError(f"{label}: LoRA A/B dimensions do not match")
            if (
                a_weight.dtype != adapter_dtype
                or a_weight.device != device
                or b_weight.dtype != adapter_dtype
                or b_weight.device != device
            ):
                raise RuntimeError(
                    f"{label}: LoRA A/B dtype and projection device must match"
                )
            if not a_weight.requires_grad or not b_weight.requires_grad:
                raise RuntimeError(f"{label}: LoRA A/B must be trainable")

    def _validate_runtime_state(self, module, module_index):
        adapter = self.adapter
        if module.disable_adapters or module.merged:
            raise RuntimeError(f"{self.label}: adapter state changed after packing")
        if tuple(module.active_adapters) != (adapter,):
            raise RuntimeError(f"{self.label}: active adapter changed after packing")
        if (
            tuple(module.lora_A.keys()) != (adapter,)
            or tuple(module.lora_B.keys()) != (adapter,)
            or tuple(module.lora_dropout.keys()) != (adapter,)
            or tuple(module.scaling.keys()) != (adapter,)
        ):
            raise RuntimeError(f"{self.label}: adapter keys changed after packing")
        if module.lora_variant:
            raise RuntimeError(f"{self.label}: LoRA variants are not supported")
        if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
            raise RuntimeError(f"{self.label}: LoRA dropout must remain zero")
        if (
            module.lora_A[adapter] is not self.a_layers[module_index]
            or module.lora_B[adapter] is not self.b_layers[module_index]
        ):
            raise RuntimeError(f"{self.label}: LoRA branch identity changed")
        a_layer = self.a_layers[module_index]
        b_layer = self.b_layers[module_index]
        if "weight" in a_layer._parameters:
            raise RuntimeError(
                f"{self.label}: packed LoRA A representation changed"
            )
        if a_layer.bias is not None or b_layer.bias is not None:
            raise RuntimeError(f"{self.label}: LoRA A/B bias changed")
        b_weight = b_layer.weight
        if b_weight is not self.b_weights[module_index]:
            raise RuntimeError(f"{self.label}: LoRA B weight identity changed")
        if (
            b_weight.shape != self.b_shapes[module_index]
            or b_weight.dtype != self.adapter_dtype
            or b_weight.device != self.device
        ):
            raise RuntimeError(
                f"{self.label}: LoRA B shape/dtype/device changed"
            )
        if (
            module.cast_input_dtype_enabled
            != self.cast_input_dtype_enabled[module_index]
        ):
            raise RuntimeError(f"{self.label}: PEFT input-cast state changed")
        base = self.base_layers[module_index]
        if (
            module.base_layer is not base
            or type(base) is not torch.nn.Linear
            or "weight" in base._parameters
            or "bias" in base._parameters
        ):
            raise RuntimeError(
                f"{self.label}: packed base representation changed"
            )

        host = self.modules[0]
        packed_weight = getattr(host, _PACKED_QKV_WEIGHT, None)
        packed_bias = getattr(host, _PACKED_QKV_BIAS, None)
        packed_a = getattr(host, _FUSED_A_PARAMETER, None)
        if (
            packed_weight is None
            or packed_weight.dtype != self.dtype
            or packed_weight.device != self.device
            or packed_weight.requires_grad
            or packed_weight.shape
            != (sum(self.output_widths), self.base_layers[0].in_features)
        ):
            raise RuntimeError(
                f"{self.label}: packed base weight dtype/device state changed"
            )
        if (
            packed_bias is None
            or packed_bias.dtype != self.dtype
            or packed_bias.device != self.device
            or packed_bias.requires_grad
            or packed_bias.shape != (sum(self.output_widths),)
        ):
            raise RuntimeError(
                f"{self.label}: packed base bias dtype/device state changed"
            )
        if packed_a is not self.packed_a:
            raise RuntimeError(f"{self.label}: packed LoRA A identity changed")
        if (
            packed_a.dtype != self.adapter_dtype
            or packed_a.device != self.device
            or packed_a.shape
            != (sum(self.ranks), self.base_layers[0].in_features)
        ):
            raise RuntimeError(
                f"{self.label}: packed LoRA A dtype/device changed"
            )
        if b_weight.requires_grad != packed_a.requires_grad:
            raise RuntimeError(
                f"{self.label}: packed LoRA A/B requires_grad states diverged"
            )
        if module_index == 0:
            _validate_packed_adapter_requires_grad(
                label=self.label,
                packed_a=packed_a,
                b_weights=self.b_weights,
            )

    def _reject_packed_state_dict(self, _module, _prefix, _keep_vars):
        raise RuntimeError(
            f"{self.label}: refusing to serialize temporary packed q/k/v "
            "projections; call materialize_standard_peft() first"
        )

    def _make_forward(self, module_index):
        group = self

        def packed_forward(module, x, *args, **kwargs):
            if args or kwargs:
                raise RuntimeError(
                    f"{group.label}: packed q/k/v accepts only the projection input"
                )
            if group._materialized:
                raise RuntimeError(
                    f"{group.label}: stale packed forward after materialization"
                )
            group._validate_runtime_state(module, module_index)
            if x.dtype != group.dtype or x.device != group.device:
                raise RuntimeError(
                    f"{group.label}: projection input dtype/device changed"
                )
            if group._next_module != module_index:
                raise RuntimeError(
                    f"{group.label}: projection call order changed "
                    f"(expected member {group._next_module}, got {module_index})"
                )

            if module_index == 0:
                if (
                    group._pending_base_pieces is not None
                    or group._pending_a_pieces is not None
                ):
                    raise RuntimeError(
                        f"{group.label}: incomplete previous packed projection"
                    )
                weight = getattr(group.modules[0], _PACKED_QKV_WEIGHT)
                bias = getattr(group.modules[0], _PACKED_QKV_BIAS)
                packed_a = getattr(group.modules[0], _FUSED_A_PARAMETER)
                projected_base = torch.nn.functional.linear(x, weight, bias)
                fused_input = module._cast_input_dtype(x, packed_a.dtype)
                projected_a = torch.nn.functional.linear(fused_input, packed_a)
                group._pending_input = x
                group._pending_base_pieces = projected_base.split(
                    group.output_widths,
                    dim=-1,
                )
                group._pending_a_pieces = projected_a.split(
                    group.ranks,
                    dim=-1,
                )
            elif x is not group._pending_input:
                raise RuntimeError(
                    f"{group.label}: q/k/v did not receive the same input tensor"
                )

            result = group._pending_base_pieces[module_index]
            result_dtype = result.dtype
            result = (
                result
                + group.b_layers[module_index](
                    group._pending_a_pieces[module_index]
                )
                * module.scaling[group.adapter]
            )
            group._next_module += 1
            if group._next_module == len(group.modules):
                group._pending_input = None
                group._pending_base_pieces = None
                group._pending_a_pieces = None
                group._next_module = 0
            return result.to(result_dtype)

        return packed_forward

    def assert_idle(self):
        if (
            self._next_module
            or self._pending_input is not None
            or self._pending_base_pieces is not None
            or self._pending_a_pieces is not None
        ):
            raise RuntimeError(
                f"{self.label}: cannot materialize during an incomplete q/k/v "
                "forward"
            )

    def materialize_standard_peft(self):
        if self._materialized:
            return
        self.assert_idle()
        host = self.modules[0]
        weight = getattr(host, _PACKED_QKV_WEIGHT)
        bias = getattr(host, _PACKED_QKV_BIAS)
        packed_a = getattr(host, _FUSED_A_PARAMETER)
        restored_weights = tuple(
            part.contiguous().clone()
            for part in weight.detach().split(self.output_widths, dim=0)
        )
        restored_biases = tuple(
            part.contiguous().clone()
            for part in bias.detach().split(self.output_widths, dim=0)
        )
        restored_a = tuple(
            part.contiguous().clone()
            for part in packed_a.detach().split(self.ranks, dim=0)
        )

        self._state_hook.remove()
        delattr(host, _PACKED_QKV_WEIGHT)
        delattr(host, _PACKED_QKV_BIAS)
        delattr(host, _FUSED_A_PARAMETER)
        for (
            module,
            base,
            a_layer,
            restored_weight,
            restored_bias,
            restored_a_weight,
            has_bias,
        ) in zip(
            self.modules,
            self.base_layers,
            self.a_layers,
            restored_weights,
            restored_biases,
            restored_a,
            self.bias_present,
        ):
            base.register_parameter(
                "weight",
                torch.nn.Parameter(restored_weight, requires_grad=False),
            )
            base.register_parameter(
                "bias",
                torch.nn.Parameter(restored_bias, requires_grad=False)
                if has_bias
                else None,
            )
            a_layer.register_parameter(
                "weight",
                torch.nn.Parameter(
                    restored_a_weight,
                    requires_grad=packed_a.requires_grad,
                ),
            )
            delattr(module, "forward")
        self.packed_a = None
        self._materialized = True


class _PackedGateUpAndLoraAGroup:
    """Pack frozen base and LoRA-A gate/up projections under one wrapper owner."""

    def __init__(self, modules, label):
        self.modules = tuple(modules)
        self.label = label
        self.adapter = tuple(self.modules[0].active_adapters)[0]
        self.base_layers = tuple(module.base_layer for module in self.modules)
        self.a_layers = tuple(module.lora_A[self.adapter] for module in self.modules)
        self.b_layers = tuple(module.lora_B[self.adapter] for module in self.modules)
        self.b_weights = tuple(layer.weight for layer in self.b_layers)
        self.output_widths = tuple(base.out_features for base in self.base_layers)
        self.ranks = tuple(layer.weight.shape[0] for layer in self.a_layers)
        self.bias_present = tuple(base.bias is not None for base in self.base_layers)
        self.b_shapes = tuple(layer.weight.shape for layer in self.b_layers)
        self.cast_input_dtype_enabled = tuple(
            module.cast_input_dtype_enabled for module in self.modules
        )
        self.dtype = self.base_layers[0].weight.dtype
        self.adapter_dtype = self.a_layers[0].weight.dtype
        self.device = self.base_layers[0].weight.device
        self._pending_input = None
        self._pending_base_pieces = None
        self._pending_a_pieces = None
        self._next_module = 0
        self._materialized = False

        packed_weight = torch.cat(
            [base.weight.detach() for base in self.base_layers],
            dim=0,
        )
        if any(self.bias_present):
            packed_bias = torch.cat(
                [
                    base.bias.detach()
                    if base.bias is not None
                    else base.weight.new_zeros(base.out_features)
                    for base in self.base_layers
                ],
                dim=0,
            )
        else:
            packed_bias = None
        packed_a = torch.nn.Parameter(
            torch.cat([layer.weight.detach() for layer in self.a_layers], dim=0),
            requires_grad=True,
        )
        self.packed_a = packed_a
        original_base_weights = tuple(base.weight for base in self.base_layers)
        original_base_biases = tuple(base.bias for base in self.base_layers)
        original_a_weights = tuple(layer.weight for layer in self.a_layers)

        host = self.modules[0]
        state_hook = None
        try:
            host.register_buffer(
                _PACKED_GATE_UP_WEIGHT,
                packed_weight,
                persistent=False,
            )
            host.register_buffer(
                _PACKED_GATE_UP_BIAS,
                packed_bias,
                persistent=False,
            )
            host.register_parameter(_FUSED_A_PARAMETER, packed_a)
            for base in self.base_layers:
                delattr(base, "weight")
                delattr(base, "bias")
            for layer in self.a_layers:
                delattr(layer, "weight")
            state_hook = host.register_state_dict_pre_hook(
                self._reject_packed_state_dict
            )
            for index, module in enumerate(self.modules):
                module.forward = MethodType(self._make_forward(index), module)
        except Exception:
            for module in self.modules:
                if "forward" in module.__dict__:
                    delattr(module, "forward")
            if state_hook is not None:
                state_hook.remove()
            if _FUSED_A_PARAMETER in host._parameters:
                delattr(host, _FUSED_A_PARAMETER)
            if _PACKED_GATE_UP_BIAS in host._buffers:
                delattr(host, _PACKED_GATE_UP_BIAS)
            if _PACKED_GATE_UP_WEIGHT in host._buffers:
                delattr(host, _PACKED_GATE_UP_WEIGHT)
            for base, weight, bias in zip(
                self.base_layers,
                original_base_weights,
                original_base_biases,
            ):
                if "weight" not in base._parameters:
                    base.register_parameter("weight", weight)
                if "bias" not in base._parameters:
                    base.register_parameter("bias", bias)
            for layer, weight in zip(self.a_layers, original_a_weights):
                if "weight" not in layer._parameters:
                    layer.register_parameter("weight", weight)
            raise
        self._state_hook = state_hook

    @staticmethod
    def validate(modules, label):
        modules = tuple(modules)
        if len(modules) != 2:
            raise RuntimeError(f"{label}: exactly gate and up modules are required")
        if len({id(module) for module in modules}) != 2:
            raise RuntimeError(f"{label}: gate and up modules must be distinct")

        active = tuple(modules[0].active_adapters)
        if len(active) != 1:
            raise RuntimeError(f"{label}: exactly one active adapter is required")
        adapter = active[0]
        first_base = modules[0].base_layer
        if type(first_base) is not torch.nn.Linear:
            raise RuntimeError(
                f"{label}: only torch.nn.Linear base layers are supported"
            )
        dtype = first_base.weight.dtype
        device = first_base.weight.device
        in_features = first_base.in_features
        out_features = first_base.out_features
        try:
            adapter_dtype = modules[0].lora_A[adapter].weight.dtype
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(f"{label}: active LoRA A weight is unavailable") from exc
        cast_input_dtype_enabled = modules[0].cast_input_dtype_enabled
        if type(cast_input_dtype_enabled) is not bool:
            raise RuntimeError(f"{label}: PEFT input-cast state must be boolean")

        for module in modules:
            if "forward" in module.__dict__:
                raise RuntimeError(
                    f"{label}: module already has an instance forward override"
                )
            if (
                hasattr(module, _PACKED_GATE_UP_WEIGHT)
                or hasattr(module, _PACKED_GATE_UP_BIAS)
                or hasattr(module, _FUSED_A_PARAMETER)
            ):
                raise RuntimeError(f"{label}: module is already projection-packed")
            if module.disable_adapters or module.merged:
                raise RuntimeError(f"{label}: adapters must be enabled and unmerged")
            if tuple(module.active_adapters) != (adapter,):
                raise RuntimeError(f"{label}: active adapters do not match")
            if (
                tuple(module.lora_A.keys()) != (adapter,)
                or tuple(module.lora_B.keys()) != (adapter,)
                or tuple(module.lora_dropout.keys()) != (adapter,)
                or tuple(module.scaling.keys()) != (adapter,)
            ):
                raise RuntimeError(
                    f"{label}: exactly one vanilla LoRA adapter key is required"
                )
            if module.lora_variant:
                raise RuntimeError(f"{label}: LoRA variants are not supported")
            if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
                raise RuntimeError(f"{label}: LoRA dropout must be zero")
            if module.cast_input_dtype_enabled != cast_input_dtype_enabled:
                raise RuntimeError(f"{label}: PEFT input-cast states do not match")

            base = module.base_layer
            if type(base) is not torch.nn.Linear:
                raise RuntimeError(
                    f"{label}: only torch.nn.Linear base layers are supported"
                )
            weight = base.weight
            if (
                weight.ndim != 2
                or base.in_features != in_features
                or base.out_features != out_features
                or weight.shape != (out_features, in_features)
            ):
                raise RuntimeError(
                    f"{label}: gate/up base projection shapes do not match"
                )
            if weight.dtype != dtype or weight.device != device:
                raise RuntimeError(
                    f"{label}: base projection dtype/device do not match"
                )
            if weight.requires_grad:
                raise RuntimeError(
                    f"{label}: base projection weights must be frozen"
                )
            if base.bias is not None:
                if (
                    base.bias.shape != (out_features,)
                    or base.bias.dtype != dtype
                    or base.bias.device != device
                ):
                    raise RuntimeError(
                        f"{label}: base bias shape/dtype/device does not match"
                    )
                if base.bias.requires_grad:
                    raise RuntimeError(
                        f"{label}: base projection biases must be frozen"
                    )

            a_layer = module.lora_A[adapter]
            b_layer = module.lora_B[adapter]
            a_weight = a_layer.weight
            b_weight = b_layer.weight
            if a_layer.bias is not None or b_layer.bias is not None:
                raise RuntimeError(f"{label}: LoRA A/B bias is not supported")
            if (
                a_weight.ndim != 2
                or a_weight.shape[1] != in_features
                or b_weight.shape != (out_features, a_weight.shape[0])
            ):
                raise RuntimeError(f"{label}: LoRA A/B dimensions do not match")
            if (
                a_weight.dtype != adapter_dtype
                or a_weight.device != device
                or b_weight.dtype != adapter_dtype
                or b_weight.device != device
            ):
                raise RuntimeError(
                    f"{label}: LoRA A/B dtype and projection device must match"
                )
            if not a_weight.requires_grad or not b_weight.requires_grad:
                raise RuntimeError(f"{label}: LoRA A/B must be trainable")

    def _validate_runtime_state(self, module, module_index):
        adapter = self.adapter
        if module.disable_adapters or module.merged:
            raise RuntimeError(f"{self.label}: adapter state changed after packing")
        if tuple(module.active_adapters) != (adapter,):
            raise RuntimeError(f"{self.label}: active adapter changed after packing")
        if (
            tuple(module.lora_A.keys()) != (adapter,)
            or tuple(module.lora_B.keys()) != (adapter,)
            or tuple(module.lora_dropout.keys()) != (adapter,)
            or tuple(module.scaling.keys()) != (adapter,)
        ):
            raise RuntimeError(f"{self.label}: adapter keys changed after packing")
        if module.lora_variant:
            raise RuntimeError(f"{self.label}: LoRA variants are not supported")
        if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
            raise RuntimeError(f"{self.label}: LoRA dropout must remain zero")
        if (
            module.lora_A[adapter] is not self.a_layers[module_index]
            or module.lora_B[adapter] is not self.b_layers[module_index]
        ):
            raise RuntimeError(f"{self.label}: LoRA branch identity changed")
        a_layer = self.a_layers[module_index]
        b_layer = self.b_layers[module_index]
        if "weight" in a_layer._parameters:
            raise RuntimeError(
                f"{self.label}: packed LoRA A representation changed"
            )
        if a_layer.bias is not None or b_layer.bias is not None:
            raise RuntimeError(f"{self.label}: LoRA A/B bias changed")
        b_weight = b_layer.weight
        if b_weight is not self.b_weights[module_index]:
            raise RuntimeError(f"{self.label}: LoRA B weight identity changed")
        if (
            b_weight.shape != self.b_shapes[module_index]
            or b_weight.dtype != self.adapter_dtype
            or b_weight.device != self.device
        ):
            raise RuntimeError(
                f"{self.label}: LoRA B shape/dtype/device changed"
            )
        if (
            module.cast_input_dtype_enabled
            != self.cast_input_dtype_enabled[module_index]
        ):
            raise RuntimeError(f"{self.label}: PEFT input-cast state changed")
        base = self.base_layers[module_index]
        if (
            module.base_layer is not base
            or type(base) is not torch.nn.Linear
            or "weight" in base._parameters
            or "bias" in base._parameters
        ):
            raise RuntimeError(
                f"{self.label}: packed base representation changed"
            )

        host = self.modules[0]
        packed_weight = getattr(host, _PACKED_GATE_UP_WEIGHT, None)
        packed_bias = getattr(host, _PACKED_GATE_UP_BIAS, None)
        packed_a = getattr(host, _FUSED_A_PARAMETER, None)
        if (
            packed_weight is None
            or packed_weight.dtype != self.dtype
            or packed_weight.device != self.device
            or packed_weight.requires_grad
            or packed_weight.shape
            != (sum(self.output_widths), self.base_layers[0].in_features)
        ):
            raise RuntimeError(
                f"{self.label}: packed base weight dtype/device state changed"
            )
        if any(self.bias_present):
            if (
                packed_bias is None
                or packed_bias.dtype != self.dtype
                or packed_bias.device != self.device
                or packed_bias.requires_grad
                or packed_bias.shape != (sum(self.output_widths),)
            ):
                raise RuntimeError(
                    f"{self.label}: packed base bias dtype/device state changed"
                )
        elif packed_bias is not None:
            raise RuntimeError(
                f"{self.label}: packed base bias representation changed"
            )
        if packed_a is not self.packed_a:
            raise RuntimeError(f"{self.label}: packed LoRA A identity changed")
        if (
            packed_a.dtype != self.adapter_dtype
            or packed_a.device != self.device
            or packed_a.shape
            != (sum(self.ranks), self.base_layers[0].in_features)
        ):
            raise RuntimeError(
                f"{self.label}: packed LoRA A dtype/device changed"
            )
        if b_weight.requires_grad != packed_a.requires_grad:
            raise RuntimeError(
                f"{self.label}: packed LoRA A/B requires_grad states diverged"
            )
        if module_index == 0:
            _validate_packed_adapter_requires_grad(
                label=self.label,
                packed_a=packed_a,
                b_weights=self.b_weights,
            )

    def _reject_packed_state_dict(self, _module, _prefix, _keep_vars):
        raise RuntimeError(
            f"{self.label}: refusing to serialize temporary packed gate/up "
            "projections; call materialize_standard_peft() first"
        )

    def _make_forward(self, module_index):
        group = self

        def packed_forward(module, x, *args, **kwargs):
            if args or kwargs:
                raise RuntimeError(
                    f"{group.label}: packed gate/up accepts only the projection input"
                )
            if group._materialized:
                raise RuntimeError(
                    f"{group.label}: stale packed forward after materialization"
                )
            group._validate_runtime_state(module, module_index)
            if x.dtype != group.dtype or x.device != group.device:
                raise RuntimeError(
                    f"{group.label}: projection input dtype/device changed"
                )
            if group._next_module != module_index:
                raise RuntimeError(
                    f"{group.label}: projection call order changed "
                    f"(expected member {group._next_module}, got {module_index})"
                )

            if module_index == 0:
                if (
                    group._pending_base_pieces is not None
                    or group._pending_a_pieces is not None
                ):
                    raise RuntimeError(
                        f"{group.label}: incomplete previous packed projection"
                    )
                weight = getattr(group.modules[0], _PACKED_GATE_UP_WEIGHT)
                bias = getattr(group.modules[0], _PACKED_GATE_UP_BIAS)
                packed_a = getattr(group.modules[0], _FUSED_A_PARAMETER)
                projected_base = torch.nn.functional.linear(x, weight, bias)
                fused_input = module._cast_input_dtype(x, packed_a.dtype)
                projected_a = torch.nn.functional.linear(fused_input, packed_a)
                group._pending_input = x
                group._pending_base_pieces = projected_base.split(
                    group.output_widths,
                    dim=-1,
                )
                group._pending_a_pieces = projected_a.split(
                    group.ranks,
                    dim=-1,
                )
            elif x is not group._pending_input:
                raise RuntimeError(
                    f"{group.label}: gate/up did not receive the same input tensor"
                )

            result = group._pending_base_pieces[module_index]
            result_dtype = result.dtype
            result = (
                result
                + group.b_layers[module_index](
                    group._pending_a_pieces[module_index]
                )
                * module.scaling[group.adapter]
            )
            group._next_module += 1
            if group._next_module == len(group.modules):
                group._pending_input = None
                group._pending_base_pieces = None
                group._pending_a_pieces = None
                group._next_module = 0
            return result.to(result_dtype)

        return packed_forward

    def assert_idle(self):
        if (
            self._next_module
            or self._pending_input is not None
            or self._pending_base_pieces is not None
            or self._pending_a_pieces is not None
        ):
            raise RuntimeError(
                f"{self.label}: cannot materialize during an incomplete gate/up "
                "forward"
            )

    def materialize_standard_peft(self):
        if self._materialized:
            return
        self.assert_idle()
        host = self.modules[0]
        weight = getattr(host, _PACKED_GATE_UP_WEIGHT)
        bias = getattr(host, _PACKED_GATE_UP_BIAS)
        packed_a = getattr(host, _FUSED_A_PARAMETER)
        restored_weights = tuple(
            part.contiguous().clone()
            for part in weight.detach().split(self.output_widths, dim=0)
        )
        if bias is None:
            restored_biases = (None,) * len(self.modules)
        else:
            restored_biases = tuple(
                part.contiguous().clone()
                for part in bias.detach().split(self.output_widths, dim=0)
            )
        restored_a = tuple(
            part.contiguous().clone()
            for part in packed_a.detach().split(self.ranks, dim=0)
        )

        self._state_hook.remove()
        delattr(host, _PACKED_GATE_UP_WEIGHT)
        delattr(host, _PACKED_GATE_UP_BIAS)
        delattr(host, _FUSED_A_PARAMETER)
        for (
            module,
            base,
            a_layer,
            restored_weight,
            restored_bias,
            restored_a_weight,
            has_bias,
        ) in zip(
            self.modules,
            self.base_layers,
            self.a_layers,
            restored_weights,
            restored_biases,
            restored_a,
            self.bias_present,
        ):
            base.register_parameter(
                "weight",
                torch.nn.Parameter(restored_weight, requires_grad=False),
            )
            base.register_parameter(
                "bias",
                torch.nn.Parameter(restored_bias, requires_grad=False)
                if has_bias
                else None,
            )
            a_layer.register_parameter(
                "weight",
                torch.nn.Parameter(
                    restored_a_weight,
                    requires_grad=packed_a.requires_grad,
                ),
            )
            delattr(module, "forward")
        self.packed_a = None
        self._materialized = True


class AllProjectionFusionPlan:
    """Install all packed projection groups as one fail-closed transaction."""

    def __init__(self, qkv_groups, gate_up_groups):
        qkv_groups = tuple(
            (tuple(modules), label) for modules, label in qkv_groups
        )
        gate_up_groups = tuple(
            (tuple(modules), label) for modules, label in gate_up_groups
        )
        if not qkv_groups or len(qkv_groups) != len(gate_up_groups):
            raise RuntimeError(
                "all fusion requires one q/k/v and gate/up group per layer"
            )

        seen = set()
        for modules, label in qkv_groups:
            _PackedQKVAndLoraAGroup.validate(modules, label)
            if seen.intersection(map(id, modules)):
                raise RuntimeError(
                    f"{label}: a module belongs to more than one fusion group"
                )
            seen.update(map(id, modules))
        for modules, label in gate_up_groups:
            _PackedGateUpAndLoraAGroup.validate(modules, label)
            if seen.intersection(map(id, modules)):
                raise RuntimeError(
                    f"{label}: a module belongs to more than one fusion group"
                )
            seen.update(map(id, modules))

        installed_qkv = []
        installed_gate_up = []
        try:
            for modules, label in qkv_groups:
                installed_qkv.append(
                    _PackedQKVAndLoraAGroup(modules, label)
                )
            for modules, label in gate_up_groups:
                installed_gate_up.append(
                    _PackedGateUpAndLoraAGroup(modules, label)
                )
        except Exception:
            for group in reversed(installed_gate_up):
                group.materialize_standard_peft()
            for group in reversed(installed_qkv):
                group.materialize_standard_peft()
            raise

        self.qkv_groups = tuple(installed_qkv)
        self.gate_up_groups = tuple(installed_gate_up)
        self._materialized = False

    def materialize_standard_peft(self):
        if self._materialized:
            return
        groups = self.qkv_groups + self.gate_up_groups
        for group in groups:
            group.assert_idle()
        for group in self.gate_up_groups:
            group.materialize_standard_peft()
        for group in self.qkv_groups:
            group.materialize_standard_peft()
        self._materialized = True


def fuse_qwen2_all_projections(transformer):
    """Fuse frozen base and LoRA-A q/k/v and gate/up projections."""
    qkv_groups = []
    gate_up_groups = []
    for index, layer in enumerate(transformer.layers):
        qkv_groups.append(
            (
                (
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ),
                f"layer {index} base+LoRA-A qkv",
            )
        )
        gate_up_groups.append(
            (
                (layer.mlp.gate_proj, layer.mlp.up_proj),
                f"layer {index} base+LoRA-A gate/up",
            )
        )
    if not qkv_groups:
        raise RuntimeError("Qwen2 transformer has no decoder layers")
    return AllProjectionFusionPlan(qkv_groups, gate_up_groups)


class _CombinedProjectionSubsetPlan:
    """Own one reviewed combined group family for a single fusion ablation."""

    def __init__(self, group_type, groups):
        normalised = tuple(
            (tuple(modules), label) for modules, label in groups
        )
        if not normalised:
            raise RuntimeError("at least one combined projection group is required")
        if group_type == "qkv":
            group_class = _PackedQKVAndLoraAGroup
        elif group_type == "gate_up":
            group_class = _PackedGateUpAndLoraAGroup
        else:
            raise RuntimeError(f"unknown combined projection group: {group_type}")

        seen = set()
        for modules, label in normalised:
            group_class.validate(modules, label)
            if seen.intersection(map(id, modules)):
                raise RuntimeError(
                    f"{label}: a module belongs to more than one fusion group"
                )
            seen.update(map(id, modules))

        installed = []
        try:
            for modules, label in normalised:
                installed.append(group_class(modules, label))
        except Exception:
            for group in reversed(installed):
                group.materialize_standard_peft()
            raise
        self.groups = tuple(installed)
        self._materialized = False

    def materialize_standard_peft(self):
        if self._materialized:
            return
        for group in self.groups:
            group.assert_idle()
        for group in reversed(self.groups):
            group.materialize_standard_peft()
        self._materialized = True


def _fuse_qwen2_combined_qkv(transformer):
    groups = [
        (
            (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ),
            f"layer {index} base+LoRA-A qkv",
        )
        for index, layer in enumerate(transformer.layers)
    ]
    return _CombinedProjectionSubsetPlan("qkv", groups)


def _fuse_qwen2_combined_gate_up(transformer):
    groups = [
        (
            (layer.mlp.gate_proj, layer.mlp.up_proj),
            f"layer {index} base+LoRA-A gate/up",
        )
        for index, layer in enumerate(transformer.layers)
    ]
    return _CombinedProjectionSubsetPlan("gate_up", groups)


def projection_strategy(features):
    """Resolve projection components without silently changing another factor."""
    return projection_components_for(features)


class _ProjectionAblationPlan:
    """Install selected disjoint helpers atomically and normalize restoration."""

    def __init__(self, components):
        self.components = tuple(components)
        self._materialized = False

    @classmethod
    def install(cls, factories):
        installed = []
        try:
            for factory, materializer_name in factories:
                installed.append((factory(), materializer_name))
        except Exception:
            for component, materializer_name in reversed(installed):
                getattr(component, materializer_name)()
            raise
        return cls(installed)

    def materialize_standard_peft(self):
        if self._materialized:
            return
        for component, materializer_name in reversed(self.components):
            getattr(component, materializer_name)()
        self._materialized = True


def install_projection_strategy(transformer, features):
    strategy = projection_strategy(features)
    if strategy == ("all_combined",):
        return fuse_qwen2_all_projections(transformer)

    factories = {
        "qkv_combined": (
            lambda: _fuse_qwen2_combined_qkv(transformer),
            "materialize_standard_peft",
        ),
        "qkv_base": (
            lambda: pack_qwen2_base_qkv(transformer),
            "materialize_standard_modules",
        ),
        "qkv_lora_a": (
            lambda: fuse_qwen2_lora_a_qkv(transformer),
            "materialize_standard_peft",
        ),
        "gate_up_combined": (
            lambda: _fuse_qwen2_combined_gate_up(transformer),
            "materialize_standard_peft",
        ),
        "gate_up_base": (
            lambda: pack_qwen2_base_gate_up(transformer),
            "materialize_standard_modules",
        ),
        "gate_up_lora_a": (
            lambda: fuse_qwen2_lora_a_gate_up(transformer),
            "materialize_standard_peft",
        ),
    }
    selected_factories = [
        factories[component]
        for component in strategy
        if component not in {"qkv_standard", "gate_up_standard"}
    ]
    return _ProjectionAblationPlan.install(selected_factories)


def read_qa(data_dir):
    """Read the saved train split. Fast path: pyarrow directly (skips importing
    `datasets`, which costs seconds); fallback: datasets.load_from_disk."""
    try:
        import pyarrow.ipc as ipc

        files = sorted(glob.glob(os.path.join(data_dir, "data-*.arrow")))
        assert files
        qs, ans = [], []
        for fp in files:
            with ipc.open_stream(fp) as reader:
                t = reader.read_all()
            qs += t.column("question").to_pylist()
            ans += t.column("answer").to_pylist()
        return qs, ans
    except Exception:
        from datasets import load_from_disk

        ds = load_from_disk(data_dir)
        return ds["question"], ds["answer"]


ANNOT_RE = re.compile(r"<<[^>]*>>")


def resolve_model_path():
    """Resolve only the exact pinned local model snapshot; never use the network."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snapshot = Path(
        hf,
        "hub",
        "models--Qwen--Qwen2.5-1.5B",
        "snapshots",
        BASE_MODEL_REVISION,
    )
    if not snapshot.is_dir():
        raise RuntimeError(
            "exact pinned Qwen2.5-1.5B snapshot is unavailable locally"
        )

    pins_path = Path(__file__).resolve().parents[2] / "harness" / "pins.json"
    if pins_path.exists():
        try:
            pins = json.loads(pins_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot authenticate local benchmark model pin") from exc
        if (
            pins.get("base_model") != BASE_MODEL
            or pins.get("base_model_sha") != BASE_MODEL_REVISION
        ):
            raise RuntimeError("local benchmark model pin does not match candidate")
    return str(snapshot)


def retire_optimizer_prefix(optimizer, active_parameters):
    """Shrink the single optimizer group and release newly retired state."""

    if len(optimizer.param_groups) != 1:
        raise RuntimeError(
            "true prefix freeze requires one optimizer parameter group"
        )
    active = tuple(active_parameters)
    active_ids = {id(parameter) for parameter in active}
    if len(active_ids) != len(active):
        raise RuntimeError("active optimizer suffix contains a duplicate parameter")

    group = optimizer.param_groups[0]
    current = tuple(group["params"])
    current_ids = {id(parameter) for parameter in current}
    if len(current_ids) != len(current):
        raise RuntimeError("optimizer parameter group contains a duplicate parameter")
    if not active_ids <= current_ids:
        raise RuntimeError(
            "optimizer parameter group does not cover the active suffix"
        )

    newly_retired = tuple(
        parameter for parameter in current if id(parameter) not in active_ids
    )
    surviving = [
        parameter for parameter in current if id(parameter) in active_ids
    ]
    if {id(parameter) for parameter in surviving} != active_ids:
        raise RuntimeError(
            "optimizer parameter order does not cover the active suffix"
        )
    group["params"] = surviving
    for parameter in newly_retired:
        optimizer.state.pop(parameter, None)
    if any(parameter in optimizer.state for parameter in newly_retired):
        raise RuntimeError("retired optimizer state was not released")
    return newly_retired


class ChunkedCE(torch.autograd.Function):
    """Cross-entropy over (hidden_states, frozen lm_head) without materializing full
    logits: per chunk, compute loss and d(loss)/d(hidden) analytically (softmax - 1)."""

    @staticmethod
    def forward(ctx, h, w, y):
        n = y.numel()
        gh = torch.empty_like(h)
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, CE_CHUNK):
            hc, yc = h[i : i + CE_CHUNK], y[i : i + CE_CHUNK]
            logits = (hc @ w.T).float()
            lse = torch.logsumexp(logits, dim=-1)
            gold = logits.gather(1, yc[:, None]).squeeze(1)
            total += (lse - gold).sum()
            logits.sub_(lse[:, None]).exp_()  # in-place softmax
            logits[torch.arange(yc.numel(), device=y.device), yc] -= 1.0
            gh[i : i + CE_CHUNK] = logits.to(h.dtype) @ w
        ctx.save_for_backward(gh)
        ctx.n = n
        return total / n

    @staticmethod
    def backward(ctx, gout):
        (gh,) = ctx.saved_tensors
        return gh * (gout / ctx.n), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    lr_config = schedule_config(TAIL_SCHEDULE_NAME)
    lr_sequence = learning_rate_sequence(TAIL_SCHEDULE_NAME)
    log(
        "production_config="
        + json.dumps(
            {
                "backward_plan": BACKWARD_PLAN_NAME,
                "engineering": ENGINEERING_VARIANT,
                "engineering_features": ENGINEERING_FEATURES.to_dict(),
                "production_features": dict(PRODUCTION_FEATURES),
                "startup": {
                    "loader": STARTUP_LOADER,
                    "page_warmer": PAGE_WARMER,
                },
                "prefix_cache": {
                    "enabled": PREFIX_CACHE_BATCH != "off",
                    "boundary_layer": 26,
                    "build_batch_blocks": PREFIX_CACHE_BATCH,
                    "dtype": "bfloat16",
                    "storage_device": "cuda",
                },
                "schedule": {
                    "epochs": EPOCHS,
                    "peak_learning_rate": LR,
                    "warmup_steps": WARMUP,
                    "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
                    **lr_config,
                    "learning_rates": lr_sequence,
                },
                "staged_backward": STAGED_PLAN.to_dict(),
            },
            sort_keys=True,
        )
    )
    if PAGE_WARMER != _STARTUP_PAGE_WARMER:
        raise RuntimeError("startup and production page-warmer controls diverged")
    seed_started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    adapter_cpu_rng_checkpoint = torch.get_rng_state().clone()
    emit_phase(
        "seed_runtime",
        started=seed_started,
        cuda_initialized=torch.cuda.is_initialized(),
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_path_started = time.monotonic()
    model_path = resolve_model_path()
    emit_phase("model_path_authentication", started=model_path_started)
    log(f"model path: {model_path}")

    auto_model_class = None
    auto_tokenizer_class = None
    direct_model_loader = None
    direct_tokenizer_loader = None
    direct_encode_batch = None
    direct_eos = None
    loader_import_started = time.monotonic()
    if STARTUP_LOADER == "auto":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        auto_model_class = AutoModelForCausalLM
        auto_tokenizer_class = AutoTokenizer
        emit_phase(
            "transformers_auto_import",
            started=loader_import_started,
        )
    else:
        from startup_loader import (
            EXPECTED_CONFIG,
            encode_text_batch,
            load_direct_qwen,
            load_direct_tokenizer,
        )

        direct_model_loader = load_direct_qwen
        direct_tokenizer_loader = load_direct_tokenizer
        direct_encode_batch = encode_text_batch
        direct_eos = EXPECTED_CONFIG["eos_token_id"]
        emit_phase(
            "direct_loader_module_import",
            started=loader_import_started,
        )

    # Load the model on a background thread while the main thread tokenizes/packs.
    holder = {}

    def _load_model():
        try:
            cuda_started = time.monotonic()
            already_initialized = torch.cuda.is_initialized()
            torch.cuda.init()
            emit_phase(
                "cuda_first_use",
                started=cuda_started,
                already_initialized=already_initialized,
                initialized=torch.cuda.is_initialized(),
            )
            load_started = time.monotonic()
            holder["model_load_started_seconds"] = load_started - T0
            if STARTUP_LOADER == "direct":
                holder["model"] = direct_model_loader(
                    model_path,
                    emit=emit_phase,
                    warmer_done=_WARMER_DONE,
                )
            else:
                holder["model"] = auto_model_class.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map="cuda",
                    local_files_only=True,
                )
            emit_phase(
                "model_construction_loading",
                started=load_started,
                loader=STARTUP_LOADER,
                warmer_complete=_WARMER_DONE.is_set(),
            )
            log("base model loaded on cuda")
        except BaseException as exc:
            holder["error"] = exc

    loader = threading.Thread(target=_load_model, name="pinned-model-loader")
    loader.start()

    tokenizer_started = time.monotonic()
    if STARTUP_LOADER == "direct":
        tok = direct_tokenizer_loader(
            model_path,
            emit=emit_phase,
        )
        eos = direct_eos
    else:
        tok = auto_tokenizer_class.from_pretrained(
            model_path,
            local_files_only=True,
        )
        eos = tok.eos_token_id
    emit_phase(
        "tokenizer_loading",
        started=tokenizer_started,
        loader=STARTUP_LOADER,
    )
    log("tokenizer ready")

    data_started = time.monotonic()
    questions, answers = read_qa(args.data_dir)
    emit_phase("training_data_loading", started=data_started, examples=len(questions))
    if STRIP_ANNOT:
        answers = [ANNOT_RE.sub("", a) for a in answers]
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    comps = [f" {a}" for a in answers]
    tokenization_started = time.monotonic()
    if STARTUP_LOADER == "direct":
        p_ids = direct_encode_batch(
            tok,
            prompts,
            emit=emit_phase,
            phase_name="prompt_tokenization",
        )
        c_ids = direct_encode_batch(
            tok,
            comps,
            emit=emit_phase,
            phase_name="completion_tokenization",
        )
    else:
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        c_ids = tok(comps, add_special_tokens=False)["input_ids"]
    emit_phase(
        "training_text_tokenization",
        started=tokenization_started,
        loader=STARTUP_LOADER,
        examples=len(questions),
    )
    log(f"tokenized {len(questions)} examples")

    packing_started = time.monotonic()
    examples = []
    for p, c in zip(p_ids, c_ids):
        c = c + [eos]
        examples.append((p + c, [-100] * len(p) + c))

    if SUBSET:
        mode, n = SUBSET.split(":")
        n = int(n)
        if mode == "shortest":
            examples.sort(key=lambda e: len(e[0]))
            examples = examples[:n]
        elif mode == "longest":
            examples.sort(key=lambda e: -len(e[0]))
            examples = examples[:n]
        elif mode == "first":
            examples = examples[:n]
        log(f"subset {SUBSET}: kept {len(examples)} examples")

    # Freeze the baseline next-fit membership, then deterministically best-fit only
    # that set. Every output bin is padded, so no attention mask is needed.
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    packing = best_fit_pack_baseline_membership(
        examples,
        capacity=PACK_LEN,
        pad_token_id=eos,
    )
    blocks_i = [list(block) for block in packing.input_blocks]
    blocks_l = [list(block) for block in packing.label_blocks]
    n_blocks = len(blocks_i)
    assert n_blocks > 0
    assert n_blocks <= packing.baseline_block_count
    assert (
        len(packing.included_source_indices)
        + len(packing.excluded_tail_source_indices)
        == len(examples)
    )
    tok_total = sum(1 for block in blocks_l for token in block if token != -100)
    log(
        "best_fit packed "
        f"{len(packing.included_source_indices)} baseline-included "
        f"examples -> {n_blocks} blocks of {PACK_LEN} ({tok_total} labeled tokens); "
        f"baseline next-fit blocks={packing.baseline_block_count}, excluded final "
        f"unflushed examples={len(packing.excluded_tail_source_indices)}"
    )
    emit_phase(
        "training_data_packing",
        started=packing_started,
        packed_blocks=n_blocks,
    )

    # The model loader exclusively owns CUDA until it has completed. This avoids
    # racing model shard H2D traffic with the packed training-data transfer.
    join_started = time.monotonic()
    loader.join()
    emit_phase("model_loader_join", started=join_started)
    if _WARMER_THREAD is not None:
        warmer_join_started = time.monotonic()
        _WARMER_THREAD.join()
        emit_phase("page_warmer_join", started=warmer_join_started)
        if (
            _WARMER_OUTCOME["status"] != "ok"
            or _WARMER_OUTCOME["success"] is not True
            or _WARMER_OUTCOME["expected_bytes"]
            != EXPECTED_MODEL_SAFETENSORS_BYTES
            or _WARMER_OUTCOME["read_bytes"]
            != EXPECTED_MODEL_SAFETENSORS_BYTES
            or _WARMER_OUTCOME["error"] is not None
        ):
            raise RuntimeError("pinned model page warmer failed closed") from None
    if "error" in holder:
        raise RuntimeError("background model loader failed") from holder["error"]
    if "model" not in holder:
        raise RuntimeError("background model loader did not return a model")
    if "model_load_started_seconds" not in holder:
        raise RuntimeError("background model loader start was not recorded")

    # CUDA must stay lazy until the model loader's measured first use. Once it is
    # initialized, reseed and checkpoint its concrete state so both loader paths
    # restore identical CPU and CUDA generators immediately before LoRA injection.
    if not torch.cuda.is_initialized():
        raise RuntimeError("CUDA was not initialized by the model loader")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("adapter RNG contract requires exactly one CUDA device")
    torch.cuda.manual_seed_all(args.seed)
    adapter_cuda_rng_checkpoint = tuple(
        state.clone() for state in torch.cuda.get_rng_state_all()
    )
    if len(adapter_cuda_rng_checkpoint) != 1:
        raise RuntimeError("initialized CUDA RNG checkpoint count differs")

    data_cuda_started = time.monotonic()
    ids_t = torch.tensor(blocks_i, dtype=torch.long, device="cuda")
    lab_t = torch.tensor(blocks_l, dtype=torch.long, device="cuda")
    emit_phase("packed_data_cuda_transfer", started=data_cuda_started)

    packed_injection = None
    if ENGINEERING_FEATURES.direct_packed_parent_runtime:
        from static_lora import inject_direct_packed_lora

        adapter_injector = inject_direct_packed_lora
    elif ENGINEERING_FEATURES.direct_static_lora:
        from static_lora import inject_static_lora

        adapter_injector = inject_static_lora
    else:
        # Keep PEFT off the direct-static import path so this factor measures both
        # injection work and its otherwise avoidable package-import overhead.
        from peft import LoraConfig, get_peft_model

        adapter_injector = None
    torch.set_rng_state(adapter_cpu_rng_checkpoint)
    torch.cuda.set_rng_state_all(adapter_cuda_rng_checkpoint)
    restored_cuda_rng_states = tuple(torch.cuda.get_rng_state_all())
    adapter_cpu_rng_restored = torch.equal(
        torch.get_rng_state(),
        adapter_cpu_rng_checkpoint,
    )
    adapter_cuda_rng_restored = (
        len(restored_cuda_rng_states) == len(adapter_cuda_rng_checkpoint)
        and all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                restored_cuda_rng_states,
                adapter_cuda_rng_checkpoint,
            )
        )
    )
    if not adapter_cpu_rng_restored or not adapter_cuda_rng_restored:
        raise RuntimeError("adapter RNG checkpoint restore failed")

    injection_started = time.monotonic()
    if ENGINEERING_FEATURES.direct_packed_parent_runtime:
        model = holder["model"]
        packed_injection = adapter_injector(
            model,
            rank=RANK,
            alpha=ALPHA,
            target_modules=TARGET_MODULES,
        )
        qwen = model
        transformer = packed_injection.transformer
    elif ENGINEERING_FEATURES.direct_static_lora:
        model = holder["model"]
        injection = adapter_injector(
            model,
            rank=RANK,
            alpha=ALPHA,
            target_modules=TARGET_MODULES,
        )
        qwen = model
        transformer = injection.transformer
    else:
        model = get_peft_model(
            holder["model"],
            LoraConfig(
                r=RANK,
                lora_alpha=ALPHA,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(TARGET_MODULES),
            ),
            autocast_adapter_dtype=False,
        )
        qwen = model.base_model.model  # Qwen2ForCausalLM with LoRA injected
        transformer = qwen.model
    model.train()
    emit_phase(
        "adapter_injection",
        started=injection_started,
        runtime=ENGINEERING_VARIANT,
    )
    dispatch_started = time.monotonic()
    if ENGINEERING_FEATURES.parent_layer_dispatch:
        from qwen_dispatch import install_parent_layer_dispatch

        projection_plan = install_parent_layer_dispatch(
            transformer,
            direct_adapter=packed_injection,
        )
    else:
        projection_plan = install_projection_strategy(
            transformer,
            PRODUCTION_FEATURES,
        )
    emit_phase("projection_dispatch_installation", started=dispatch_started)
    staged_backward = StagedBackwardTruncation(
        transformer.layers,
        STAGED_PLAN,
        retire_prefix_parameters=ENGINEERING_FEATURES.true_prefix_freeze,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    staged_backward.validate_trainable_partition(trainable)
    log(f"trainable params: {n_trainable:,}")
    if n_trainable != EXPECTED_SAVED_ADAPTER_PARAMETERS:
        raise RuntimeError(
            "static adapter parameter count changed: "
            f"expected {EXPECTED_SAVED_ADAPTER_PARAMETERS:,}, got {n_trainable:,}"
        )
    if any(parameter.dtype != torch.bfloat16 for parameter in trainable):
        raise RuntimeError(
            "resolved adapter dtype does not match every trainable parameter"
        )

    lm_w = qwen.lm_head.weight  # frozen (tied) — ChunkedCE never trains it

    optimizer_started = time.monotonic()
    opt = torch.optim.AdamW(
        trainable,
        lr=LR,
        betas=(0.9, ADAM_B2),
        weight_decay=0.0,
        fused=True,
    )
    emit_phase("optimizer_construction", started=optimizer_started)

    horizon_started = time.monotonic()
    horizon = build_training_horizon(
        n_blocks,
        EPOCHS,
        BS,
        MAX_OPTIMIZER_STEPS,
    )
    total_blocks = horizon.total_blocks
    steps = horizon.optimizer_steps
    if steps != len(lr_sequence) or steps != MAX_OPTIMIZER_STEPS:
        raise RuntimeError("packed data did not satisfy the precommitted LR horizon")
    staged_backward.validate_total_steps(steps)
    order = build_shuffled_block_order(horizon, rng)
    if len(order) != total_blocks:
        raise RuntimeError("seeded block order differs from the training horizon")
    emit_phase("training_horizon_construction", started=horizon_started)

    if PRODUCTION_FEATURES["contiguous_hotloop"]:
        # This is the same advanced-indexing row selection the unfused loop performs
        # in every step, concatenated into one operation. Later batches are views.
        epoch_plan = build_epoch_plan(order, blocks_l, BS)
        if len(epoch_plan.batches) != steps:
            raise RuntimeError("contiguous plan changed the optimizer-step count")
        epoch_ids_t = ids_t[list(epoch_plan.order)]
        epoch_lab_t = lab_t[list(epoch_plan.order)]
        if not epoch_ids_t.is_contiguous() or not epoch_lab_t.is_contiguous():
            raise RuntimeError(
                "one-time advanced indexing did not produce contiguous tensors"
            )
        del ids_t, lab_t

        # Precompute the exact completion-only flattening once. These tensors are
        # copies, so labels can be released before the first optimizer step.
        completion_batches = []
        for batch in epoch_plan.batches:
            labels_view = epoch_lab_t[batch.start : batch.stop]
            shifted_labels = labels_view[:, 1:].reshape(-1)
            completion_positions = (shifted_labels != -100).nonzero(
                as_tuple=True
            )[0]
            completion_targets = shifted_labels[completion_positions]
            if completion_positions.numel() != len(batch.completion_positions):
                raise RuntimeError(
                    "completion layout differs from the exact CPU proof plan"
                )
            completion_batches.append(
                (completion_positions, completion_targets)
            )
        del epoch_lab_t
    elif PREFIX_CACHE_BATCH != "off":
        raise RuntimeError("prefix caching requires the contiguous hot loop")

    def lr_at(step):
        return learning_rate_at(TAIL_SCHEDULE_NAME, step)

    log(
        "training_horizon="
        + json.dumps(
            horizon.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    log(
        f"training: {steps} steps over {total_blocks} blocks, "
        f"lr={LR}, effective_epochs={horizon.effective_epoch_fraction:.6f}"
    )
    t_train = time.monotonic()
    full_stage_started = torch.cuda.Event(enable_timing=True)
    full_stage_finished = torch.cuda.Event(enable_timing=True)
    tail_stage_started = torch.cuda.Event(enable_timing=True)
    tail_stage_finished = torch.cuda.Event(enable_timing=True)
    full_stage_started.record()
    prefix_runner = None
    prefix_metrics = None
    stage_metrics = None
    tail_block_offset = FULL_BACKWARD_UPDATES * BS
    has_tail = steps > FULL_BACKWARD_UPDATES

    def apply_staged_transition(step):
        transition = staged_backward.transition_before_step(step)
        if transition is not None:
            if ENGINEERING_FEATURES.true_prefix_freeze:
                retire_optimizer_prefix(
                    opt,
                    staged_backward.upper_parameters,
                )
            log(
                "staged backward transition="
                + json.dumps(
                    {
                        **transition.to_dict(),
                        "saved_adapter_params": n_trainable,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return transition

    def parameters_to_clip():
        if (
            ENGINEERING_FEATURES.true_prefix_freeze
            and staged_backward.active
        ):
            return staged_backward.upper_parameters
        return trainable

    if PRODUCTION_FEATURES["contiguous_hotloop"]:
        for step, batch in enumerate(epoch_plan.batches):
            transition = apply_staged_transition(step)
            if step == FULL_BACKWARD_UPDATES:
                full_stage_finished.record()
                if transition is None:
                    raise RuntimeError("prefix cache boundary transition was not applied")
                if batch.start != tail_block_offset:
                    raise RuntimeError(
                        "tail block offset differs from configured full updates"
                    )
                if PREFIX_CACHE_BATCH != "off":
                    from frozen_prefix import FrozenPrefixTailRunner

                    cache_started = time.monotonic()
                    build_batch = (
                        "all"
                        if PREFIX_CACHE_BATCH == "all"
                        else int(PREFIX_CACHE_BATCH)
                    )
                    prefix_runner = FrozenPrefixTailRunner(
                        transformer=transformer,
                        boundary_index=staged_backward.boundary_index,
                        sequence_length=PACK_LEN,
                        materialization_batch_blocks=build_batch,
                    )
                    prefix_metrics = prefix_runner.materialize(
                        epoch_ids_t[tail_block_offset:]
                    )
                    emit_phase(
                        "frozen_prefix_materialization",
                        started=cache_started,
                        **prefix_metrics.to_dict(),
                    )
                    log(
                        "prefix_cache_metrics="
                        + json.dumps(
                            prefix_metrics.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                tail_stage_started.record()
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            x = epoch_ids_t[batch.start : batch.stop]
            completion_positions, completion_targets = completion_batches[step]
            if prefix_runner is None:
                h = transformer(input_ids=x, use_cache=False).last_hidden_state
            elif step < FULL_BACKWARD_UPDATES:
                raise RuntimeError("prefix cache was materialized before its boundary")
            else:
                cache_start = batch.start - tail_block_offset
                cache_stop = batch.stop - tail_block_offset
                h = prefix_runner.forward_suffix(cache_start, cache_stop)
            hs = h[:, :-1, :].reshape(-1, h.shape[-1])
            loss = ChunkedCE.apply(
                hs[completion_positions], lm_w, completion_targets
            )
            loss.backward()
            staged_backward.assert_gradient_partition()
            torch.nn.utils.clip_grad_norm_(parameters_to_clip(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 10 == 0 and step != steps - 1:
                log(f"step {step + 1}/{steps} queued lr={lr_at(step):.2e}")
            if step == steps - 1:
                if has_tail:
                    tail_stage_finished.record()
                    tail_stage_finished.synchronize()
                    tail_stage_seconds = (
                        tail_stage_started.elapsed_time(
                            tail_stage_finished
                        )
                        / 1000.0
                    )
                else:
                    full_stage_finished.record()
                    full_stage_finished.synchronize()
                    tail_stage_seconds = 0.0
                final_loss = loss.item()
                dt = time.monotonic() - t_train
                processed_blocks = horizon.processed_blocks_after(step)
                log(
                    f"step {step + 1}/{steps} loss={final_loss:.4f} "
                    f"lr={lr_at(step):.2e} "
                    f"({processed_blocks * PACK_LEN / max(dt, 1e-9):,.0f} tok/s)"
                )
                stage_metrics = {
                    "full_stage_cuda_seconds": (
                        full_stage_started.elapsed_time(full_stage_finished) / 1000.0
                    ),
                    "prefix_cache_build_seconds": (
                        0.0
                        if prefix_metrics is None
                        else prefix_metrics.build_seconds
                    ),
                    "tail_stage_cuda_seconds": tail_stage_seconds,
                    "training_loop_wall_seconds": dt,
                }
                log(
                    "training_stage_metrics="
                    + json.dumps(
                        stage_metrics,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
    else:
        # Exact historical hot loop: advanced-index rows and completion labels on
        # every step instead of materializing one contiguous epoch plan.
        for step in range(steps):
            apply_staged_transition(step)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            selected_rows = order[step * BS : (step + 1) * BS]
            x = ids_t[selected_rows]
            y = lab_t[selected_rows]
            h = transformer(input_ids=x, use_cache=False).last_hidden_state
            hs = h[:, :-1, :].reshape(-1, h.shape[-1])
            shifted_labels = y[:, 1:].reshape(-1)
            completion_positions = (shifted_labels != -100).nonzero(
                as_tuple=True
            )[0]
            completion_targets = shifted_labels[completion_positions]
            loss = ChunkedCE.apply(
                hs[completion_positions], lm_w, completion_targets
            )
            loss.backward()
            staged_backward.assert_gradient_partition()
            torch.nn.utils.clip_grad_norm_(parameters_to_clip(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 10 == 0 and step != steps - 1:
                log(f"step {step + 1}/{steps} queued lr={lr_at(step):.2e}")
            if step == steps - 1:
                final_loss = loss.item()
                dt = time.monotonic() - t_train
                processed_blocks = horizon.processed_blocks_after(step)
                log(
                    f"step {step + 1}/{steps} loss={final_loss:.4f} "
                    f"lr={lr_at(step):.2e} "
                    f"({processed_blocks * PACK_LEN / max(dt, 1e-9):,.0f} tok/s)"
                )

    if stage_metrics is None:
        raise RuntimeError("training stage metrics were not finalized")
    if PREFIX_CACHE_BATCH == "off" and prefix_runner is not None:
        raise RuntimeError("prefix cache ran while configured off")
    if PREFIX_CACHE_BATCH != "off" and prefix_runner is None:
        raise RuntimeError("configured prefix cache was not materialized")
    if prefix_runner is not None:
        prefix_runner.close()

    # Restore ordinary parent forwards and frozen Linear storage. Wrapper-backed
    # variants also restore their standard LoRA-A modules; the direct-packed
    # variant stays descriptor-backed while its writer emits only canonical PEFT
    # keys. Every exported adapter uses the stock loader.
    staged_backward.assert_completed()
    log(
        "staged_backward_summary="
        + json.dumps(
            staged_backward.summary(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    staged_backward.restore_for_export()
    staged_backward.close()
    export_started = time.monotonic()
    projection_plan.materialize_standard_peft()
    if ENGINEERING_FEATURES.direct_adapter_writer:
        from adapter_writer import write_static_lora_adapter

        write_static_lora_adapter(
            transformer=transformer,
            output_dir=args.output_dir,
            base_model_name_or_path=model_path,
            rank=RANK,
            alpha=ALPHA,
            target_modules=TARGET_MODULES,
            packed_injection=packed_injection,
        )
    else:
        from adapter_writer import normalize_peft_artifact

        model.save_pretrained(args.output_dir)
        normalize_peft_artifact(args.output_dir)
    emit_phase("adapter_export", started=export_started)
    log(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
