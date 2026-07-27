"""Track 2 port of 004-slippylolo: BF16 LoRA with best-fit packing for SmolLM2/SQuAD.

What's different from the Track 2 baseline (t2-000-baseline):
  1. A 0.75-epoch schedule at 5e-4 with 6 warmup steps.
  2. Shortest-4,000 example subset by tokenized prompt+answer length.
  3. No HF Trainer: a hand-rolled loop over pre-packed fixed-length GPU-resident blocks
     (zero dataloader/collate overhead, no attention mask needed — all blocks are full).
  4. Chunked completion-only cross-entropy as a custom autograd.Function: lm_head logits
     are never materialized for the whole batch, and are only computed for the ~30% of
     tokens that carry labels.
  5. LoRA parameters stay in the base model's BF16 dtype instead of PEFT's default FP32
     autocast, avoiding full-size FP32 adapter saves and per-layer dtype casts.
  6. Deterministic best-fit repacks exactly the examples emitted by the baseline's
     next-fit loop.
  7. Frozen q/k/v base projections and their shared-input LoRA A projections each
     execute as one packed GEMM per layer during training.
  8. Frozen gate/up base projections and their shared-input LoRA A projections each
     execute as one packed GEMM per layer during training.
  9. Model safetensors are read into the page cache on a background thread while
     torch/transformers import, overlapping the two biggest fixed costs.

Contract: python train.py --data-dir <squad_train> --output-dir <dir> --seed <int>
"""

import argparse
import glob
import os
import threading
import time

T0 = time.monotonic()


def log(msg):
    print(f"[t+{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B"
BASE_MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"


def _warm_model_files():
    """Pull the model shards into the OS page cache while python imports torch."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for pat in (f"{hf}/hub/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/*/*.safetensors",):
        for p in glob.glob(pat):
            try:
                with open(p, "rb") as f:
                    while f.read(1 << 25):
                        pass
            except OSError:
                pass


threading.Thread(target=_warm_model_files, daemon=True).start()

import json  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402
from types import MethodType  # noqa: E402

import torch  # noqa: E402
from hot_loop_plan import build_epoch_plan  # noqa: E402
from packing import best_fit_pack_baseline_membership  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402


_PACKED_QKV_WEIGHT = "_packed_base_qkv_weight"
_PACKED_QKV_BIAS = "_packed_base_qkv_bias"
_PACKED_GATE_UP_WEIGHT = "_packed_base_gate_up_weight"
_PACKED_GATE_UP_BIAS = "_packed_base_gate_up_bias"
_FUSED_A_PARAMETER = "_packed_a_weight"


class _PackedQKVAndLoraAGroup:
    """Pack frozen base and LoRA-A q/k/v projections under one wrapper owner."""

    def __init__(self, modules, label):
        self.modules = tuple(modules)
        self.label = label
        self.adapter = tuple(self.modules[0].active_adapters)[0]
        self.base_layers = tuple(module.base_layer for module in self.modules)
        self.a_layers = tuple(module.lora_A[self.adapter] for module in self.modules)
        self.b_layers = tuple(module.lora_B[self.adapter] for module in self.modules)
        self.output_widths = tuple(base.out_features for base in self.base_layers)
        self.ranks = tuple(layer.weight.shape[0] for layer in self.a_layers)
        self.bias_present = tuple(base.bias is not None for base in self.base_layers)
        self.b_shapes = tuple(layer.weight.shape for layer in self.b_layers)
        self.cast_input_dtype_enabled = tuple(
            module.cast_input_dtype_enabled for module in self.modules
        )
        self.dtype = self.base_layers[0].weight.dtype
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
                a_weight.dtype != dtype
                or a_weight.device != device
                or b_weight.dtype != dtype
                or b_weight.device != device
            ):
                raise RuntimeError(
                    f"{label}: base and LoRA A/B dtype/device must match"
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
        if (
            b_layer.weight.shape != self.b_shapes[module_index]
            or b_layer.weight.dtype != self.dtype
            or b_layer.weight.device != self.device
            or not b_layer.weight.requires_grad
        ):
            raise RuntimeError(
                f"{self.label}: LoRA B shape/dtype/device state changed"
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
        if (
            packed_a is None
            or packed_a.dtype != self.dtype
            or packed_a.device != self.device
            or not packed_a.requires_grad
            or packed_a.shape
            != (sum(self.ranks), self.base_layers[0].in_features)
        ):
            raise RuntimeError(
                f"{self.label}: packed LoRA A dtype/device state changed"
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
        self.output_widths = tuple(base.out_features for base in self.base_layers)
        self.ranks = tuple(layer.weight.shape[0] for layer in self.a_layers)
        self.bias_present = tuple(base.bias is not None for base in self.base_layers)
        self.b_shapes = tuple(layer.weight.shape for layer in self.b_layers)
        self.cast_input_dtype_enabled = tuple(
            module.cast_input_dtype_enabled for module in self.modules
        )
        self.dtype = self.base_layers[0].weight.dtype
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
                a_weight.dtype != dtype
                or a_weight.device != device
                or b_weight.dtype != dtype
                or b_weight.device != device
            ):
                raise RuntimeError(
                    f"{label}: base and LoRA A/B dtype/device must match"
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
        if (
            b_layer.weight.shape != self.b_shapes[module_index]
            or b_layer.weight.dtype != self.dtype
            or b_layer.weight.device != self.device
            or not b_layer.weight.requires_grad
        ):
            raise RuntimeError(
                f"{self.label}: LoRA B shape/dtype/device state changed"
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
        if (
            packed_a is None
            or packed_a.dtype != self.dtype
            or packed_a.device != self.device
            or not packed_a.requires_grad
            or packed_a.shape
            != (sum(self.ranks), self.base_layers[0].in_features)
        ):
            raise RuntimeError(
                f"{self.label}: packed LoRA A dtype/device state changed"
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


def fuse_smollm2_all_projections(transformer):
    """Fuse frozen base and LoRA-A q/k/v and gate/up projections for SmolLM2."""
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
        raise RuntimeError("SmolLM2 (Llama) model has no decoder layers")
    return AllProjectionFusionPlan(qkv_groups, gate_up_groups)


def read_squad(data_dir):
    """Read the saved SQuAD train split. Fast path: pyarrow directly; fallback: datasets."""
    try:
        import pyarrow.ipc as ipc

        files = sorted(glob.glob(os.path.join(data_dir, "data-*.arrow")))
        assert files
        contexts, questions, answers_list = [], [], []
        for fp in files:
            with ipc.open_stream(fp) as reader:
                t = reader.read_all()
            contexts += t.column("context").to_pylist()
            questions += t.column("question").to_pylist()
            answers_list += t.column("answers").to_pylist()
        return contexts, questions, answers_list
    except Exception:
        from datasets import load_from_disk

        ds = load_from_disk(data_dir)
        return ds["context"], ds["question"], ds["answers"]


# --- iteration knobs (env-overridable; defaults are the submitted config) ---
LR = float(os.environ.get("SR_LR", "5e-4"))
EPOCHS = float(os.environ.get("SR_EPOCHS", "0.75"))
BS = int(os.environ.get("SR_BS", "8"))
PACK_LEN = int(os.environ.get("SR_PACK", "1024"))
WARMUP = int(os.environ.get("SR_WARMUP", "6"))
MIN_LR_FRAC = float(os.environ.get("SR_MIN_LR_FRAC", "0.05"))
RANK = int(os.environ.get("SR_RANK", "16"))
ALPHA = int(os.environ.get("SR_ALPHA", "32"))
ADAM_B2 = float(os.environ.get("SR_ADAM_B2", "0.95"))
SUBSET = os.environ.get("SR_SUBSET", "shortest:4000")
CE_CHUNK = int(os.environ.get("SR_CE_CHUNK", "2048"))


def resolve_model_path():
    """Resolve only the exact pinned local model snapshot; never use the network."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snapshot = Path(
        hf,
        "hub",
        "models--HuggingFaceTB--SmolLM2-1.7B",
        "snapshots",
        BASE_MODEL_REVISION,
    )
    if not snapshot.is_dir():
        raise RuntimeError(
            "exact pinned SmolLM2-1.7B snapshot is unavailable locally"
        )

    pins_path = Path(__file__).resolve().parents[2] / "harness" / "pins-t2.json"
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

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_path = resolve_model_path()
    log(f"model path: {model_path}")

    # Load the model on a background thread while the main thread tokenizes/packs.
    holder = {}

    def _load_model():
        holder["model"] = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
            local_files_only=True,
        )
        log("base model loaded on cuda")

    loader = threading.Thread(target=_load_model)
    loader.start()

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    log("tokenizer ready")

    contexts, questions, answers_list = read_squad(args.data_dir)
    prompts = [f"Context: {c}\nQuestion: {q}\nAnswer:" for c, q in zip(contexts, questions)]
    comps = [f" {a['text'][0]}" if isinstance(a, dict) else f" {a}" for a in answers_list]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(comps, add_special_tokens=False)["input_ids"]
    log(f"tokenized {len(contexts)} examples")

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
        f"best-fit packed {len(packing.included_source_indices)} baseline-included "
        f"examples -> {n_blocks} blocks of {PACK_LEN} ({tok_total} labeled tokens); "
        f"baseline next-fit blocks={packing.baseline_block_count}, excluded final "
        f"unflushed examples={len(packing.excluded_tail_source_indices)}"
    )

    ids_t = torch.tensor(blocks_i, dtype=torch.long, device="cuda")
    lab_t = torch.tensor(blocks_l, dtype=torch.long, device="cuda")

    loader.join()
    model = get_peft_model(
        holder["model"],
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
            ],
        ),
        autocast_adapter_dtype=False,
    )
    model.train()
    smollm = model.base_model.model
    transformer = smollm.model
    all_fusion = fuse_smollm2_all_projections(transformer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log(f"trainable params: {n_trainable:,}")
    assert n_trainable <= 30_000_000

    lm_w = smollm.lm_head.weight  # frozen (tied) — ChunkedCE never trains it

    opt = torch.optim.AdamW(
        trainable,
        lr=LR,
        betas=(0.9, ADAM_B2),
        weight_decay=0.0,
        fused=True,
    )

    total_blocks = round(n_blocks * EPOCHS)
    steps = (total_blocks + BS - 1) // BS
    order = []
    while len(order) < total_blocks:
        ep = list(range(n_blocks))
        rng.shuffle(ep)
        order += ep
    order = order[:total_blocks]

    # This is the same advanced-indexing row selection the unfused loop performs in
    # every step, concatenated into one operation. Later batches are contiguous views.
    epoch_plan = build_epoch_plan(order, blocks_l, BS)
    if len(epoch_plan.batches) != steps:
        raise RuntimeError("contiguous plan changed the optimizer-step count")
    epoch_ids_t = ids_t[list(epoch_plan.order)]
    epoch_lab_t = lab_t[list(epoch_plan.order)]
    if not epoch_ids_t.is_contiguous() or not epoch_lab_t.is_contiguous():
        raise RuntimeError("one-time advanced indexing did not produce contiguous tensors")
    del ids_t, lab_t

    # Precompute the exact completion-only flattening once. These tensors are copies,
    # so the permuted label tensor can be released before the first optimizer step.
    completion_batches = []
    for batch in epoch_plan.batches:
        labels_view = epoch_lab_t[batch.start : batch.stop]
        shifted_labels = labels_view[:, 1:].reshape(-1)
        completion_positions = (shifted_labels != -100).nonzero(as_tuple=True)[0]
        completion_targets = shifted_labels[completion_positions]
        if completion_positions.numel() != len(batch.completion_positions):
            raise RuntimeError("completion layout differs from the exact CPU proof plan")
        completion_batches.append((completion_positions, completion_targets))
    del epoch_lab_t

    def lr_at(step):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        import math

        f = (step - WARMUP) / max(1, steps - WARMUP)
        return LR * (
            MIN_LR_FRAC
            + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * f))
        )

    log(f"training: {steps} steps x {BS} blocks, lr={LR}, epochs={EPOCHS}")
    t_train = time.monotonic()
    for step, batch in enumerate(epoch_plan.batches):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x = epoch_ids_t[batch.start : batch.stop]
        completion_positions, completion_targets = completion_batches[step]
        h = transformer(input_ids=x, use_cache=False).last_hidden_state
        hs = h[:, :-1, :].reshape(-1, h.shape[-1])
        loss = ChunkedCE.apply(
            hs[completion_positions], lm_w, completion_targets
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 and step != steps - 1:
            log(f"step {step + 1}/{steps} queued lr={lr_at(step):.2e}")
        if step == steps - 1:
            final_loss = loss.item()
            dt = time.monotonic() - t_train
            log(
                f"step {step + 1}/{steps} loss={final_loss:.4f} "
                f"lr={lr_at(step):.2e} "
                f"({batch.stop * PACK_LEN / max(dt, 1e-9):,.0f} tok/s)"
            )

    # Restore ordinary frozen Linear parameters, LoRA A parameters, and PEFT wrapper
    # forwards before save.
    all_fusion.materialize_standard_peft()
    model.save_pretrained(args.output_dir)
    log(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
