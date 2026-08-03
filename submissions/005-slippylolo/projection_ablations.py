"""Reviewed projection helpers used only by reverse-ablation paths.

The helper bodies below are copied from the repository's exact historical isolated
candidates. Keeping them outside train.py makes the winning default path auditable.
"""

from types import MethodType

import torch

_PACKED_QKV_WEIGHT = "_packed_base_qkv_weight"
_PACKED_QKV_BIAS = "_packed_base_qkv_bias"


class _PackedBaseQKVGroup:
    """Use one frozen base projection for adjacent PEFT q/k/v wrappers.

    LoRA A/B parameters and computations remain independent and unchanged. Only the
    frozen ``base_layer`` weights are represented temporarily as row-concatenated,
    non-persistent buffers. The original Linear modules stay in place and are fully
    materialized again before serialization.
    """

    def __init__(self, modules, label):
        self.modules = tuple(modules)
        self.label = label
        self.adapter = tuple(self.modules[0].active_adapters)[0]
        self.base_layers = tuple(module.base_layer for module in self.modules)
        self.output_widths = tuple(base.weight.shape[0] for base in self.base_layers)
        self.bias_present = tuple(base.bias is not None for base in self.base_layers)
        self._pending_input = None
        self._pending_pieces = None
        self._next_module = 0
        self._materialized = False

        packed_weight = torch.cat([base.weight.detach() for base in self.base_layers], dim=0)
        bias_parts = [
            base.bias.detach() if base.bias is not None else base.weight.new_zeros(base.weight.shape[0])
            for base in self.base_layers
        ]
        packed_bias = torch.cat(bias_parts, dim=0)

        host = self.modules[0]
        host.register_buffer(_PACKED_QKV_WEIGHT, packed_weight, persistent=False)
        host.register_buffer(_PACKED_QKV_BIAS, packed_bias, persistent=False)
        for base in self.base_layers:
            delattr(base, "weight")
            delattr(base, "bias")

        self._state_hook = host.register_state_dict_pre_hook(self._reject_packed_state_dict)
        for index, module in enumerate(self.modules):
            module.forward = MethodType(self._make_forward(index), module)

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
        first_weight = modules[0].base_layer.weight
        in_features = first_weight.shape[1]

        for module in modules:
            if "forward" in module.__dict__:
                raise RuntimeError(f"{label}: module already has an instance forward override")
            if hasattr(module, _PACKED_QKV_WEIGHT) or hasattr(module, _PACKED_QKV_BIAS):
                raise RuntimeError(f"{label}: module is already base-packed")
            if module.disable_adapters or module.merged:
                raise RuntimeError(f"{label}: adapters must be enabled and unmerged")
            if tuple(module.active_adapters) != (adapter,):
                raise RuntimeError(f"{label}: active adapters do not match")
            if tuple(module.lora_A.keys()) != (adapter,) or tuple(module.lora_B.keys()) != (adapter,):
                raise RuntimeError(f"{label}: exactly one vanilla LoRA A/B pair is required")
            if adapter in module.lora_variant:
                raise RuntimeError(f"{label}: LoRA variants are not supported")
            if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
                raise RuntimeError(f"{label}: LoRA dropout must be zero")
            if module.lora_A[adapter].bias is not None or module.lora_B[adapter].bias is not None:
                raise RuntimeError(f"{label}: LoRA A/B bias is not supported")

            base = module.base_layer
            weight = base.weight
            if weight.ndim != 2 or weight.shape[1] != in_features:
                raise RuntimeError(f"{label}: base projection input dimensions do not match")
            if weight.dtype != first_weight.dtype or weight.device != first_weight.device:
                raise RuntimeError(f"{label}: base projection dtype/device do not match")
            if weight.requires_grad:
                raise RuntimeError(f"{label}: base projection weights must be frozen")
            if base.bias is not None:
                if base.bias.dtype != weight.dtype or base.bias.device != weight.device:
                    raise RuntimeError(f"{label}: base bias dtype/device does not match its weight")
                if base.bias.requires_grad:
                    raise RuntimeError(f"{label}: base projection biases must be frozen")

    def _reject_packed_state_dict(self, _module, _prefix, _keep_vars):
        raise RuntimeError(
            f"{self.label}: refusing to serialize temporary packed base q/k/v weights; "
            "call materialize_standard_modules() first"
        )

    def _make_forward(self, module_index):
        group = self

        def packed_forward(module, x, *args, **kwargs):
            if args or kwargs:
                raise RuntimeError(f"{group.label}: packed q/k/v accepts only the projection input")
            if group._materialized:
                raise RuntimeError(f"{group.label}: stale packed forward after materialization")
            if module.disable_adapters or module.merged:
                raise RuntimeError(f"{group.label}: adapter state changed after packing")
            if tuple(module.active_adapters) != (group.adapter,):
                raise RuntimeError(f"{group.label}: active adapter changed after packing")
            if group._next_module != module_index:
                raise RuntimeError(
                    f"{group.label}: projection call order changed "
                    f"(expected member {group._next_module}, got {module_index})"
                )

            if module_index == 0:
                if group._pending_pieces is not None:
                    raise RuntimeError(f"{group.label}: incomplete previous packed projection")
                weight = getattr(group.modules[0], _PACKED_QKV_WEIGHT)
                bias = getattr(group.modules[0], _PACKED_QKV_BIAS)
                projected = torch.nn.functional.linear(x, weight, bias)
                group._pending_input = x
                group._pending_pieces = projected.split(group.output_widths, dim=-1)
            elif x is not group._pending_input:
                raise RuntimeError(f"{group.label}: q/k/v did not receive the same input tensor")

            result = group._pending_pieces[module_index]
            result_dtype = result.dtype
            lora_a = module.lora_A[group.adapter]
            lora_b = module.lora_B[group.adapter]
            lora_input = module._cast_input_dtype(x, lora_a.weight.dtype)
            result = result + lora_b(module.lora_dropout[group.adapter](lora_a(lora_input))) * module.scaling[
                group.adapter
            ]

            group._next_module += 1
            if group._next_module == len(group.modules):
                group._pending_input = None
                group._pending_pieces = None
                group._next_module = 0
            return result.to(result_dtype)

        return packed_forward

    def assert_idle(self):
        if self._next_module or self._pending_input is not None or self._pending_pieces is not None:
            raise RuntimeError(f"{self.label}: cannot materialize during an incomplete q/k/v forward")

    def materialize_standard_modules(self):
        if self._materialized:
            return
        self.assert_idle()
        host = self.modules[0]
        weight = getattr(host, _PACKED_QKV_WEIGHT)
        bias = getattr(host, _PACKED_QKV_BIAS)
        weight_parts = weight.detach().split(self.output_widths, dim=0)
        bias_parts = bias.detach().split(self.output_widths, dim=0)

        self._state_hook.remove()
        delattr(host, _PACKED_QKV_WEIGHT)
        delattr(host, _PACKED_QKV_BIAS)
        for module, base, weight_part, bias_part, has_bias in zip(
            self.modules,
            self.base_layers,
            weight_parts,
            bias_parts,
            self.bias_present,
        ):
            base.register_parameter(
                "weight",
                torch.nn.Parameter(weight_part.contiguous().clone(), requires_grad=False),
            )
            base.register_parameter(
                "bias",
                torch.nn.Parameter(bias_part.contiguous().clone(), requires_grad=False) if has_bias else None,
            )
            delattr(module, "forward")
        self._materialized = True


class PackedBaseQKVPlan:
    """Own packed attention groups and restore the ordinary module tree."""

    def __init__(self, groups):
        normalised = tuple((tuple(modules), label) for modules, label in groups)
        seen = set()
        for modules, label in normalised:
            _PackedBaseQKVGroup.validate(modules, label)
            if seen.intersection(map(id, modules)):
                raise RuntimeError(f"{label}: a module belongs to more than one packed group")
            seen.update(map(id, modules))

        installed = []
        try:
            for modules, label in normalised:
                installed.append(_PackedBaseQKVGroup(modules, label))
        except Exception:
            for group in installed:
                group.materialize_standard_modules()
            raise
        self.groups = tuple(installed)
        self._materialized = False

    def materialize_standard_modules(self):
        if self._materialized:
            return
        for group in self.groups:
            group.assert_idle()
        for group in self.groups:
            group.materialize_standard_modules()
        self._materialized = True


def pack_qwen2_base_qkv(transformer):
    """Pack frozen q/k/v base projections in every Qwen2 decoder layer."""
    groups = [
        (
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
            f"layer {index} base qkv",
        )
        for index, layer in enumerate(transformer.layers)
    ]
    if not groups:
        raise RuntimeError("Qwen2 transformer has no decoder layers")
    return PackedBaseQKVPlan(groups)




_PACKED_GATE_UP_WEIGHT = "_packed_base_gate_up_weight"
_PACKED_GATE_UP_BIAS = "_packed_base_gate_up_bias"


class _PackedBaseGateUpGroup:
    """Use one frozen base projection for adjacent PEFT gate/up wrappers.

    The two LoRA A/B branches remain independent and unchanged. Only the frozen
    base-layer weights are represented temporarily as a row-concatenated,
    non-persistent buffer. The original Linear modules remain in the model tree and
    are fully materialized again before serialization.
    """

    def __init__(self, modules, label):
        self.modules = tuple(modules)
        self.label = label
        self.adapter = tuple(self.modules[0].active_adapters)[0]
        self.base_layers = tuple(module.base_layer for module in self.modules)
        self.output_widths = tuple(base.weight.shape[0] for base in self.base_layers)
        self.bias_present = tuple(base.bias is not None for base in self.base_layers)
        self._pending_input = None
        self._pending_up = None
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
                    else base.weight.new_zeros(base.weight.shape[0])
                    for base in self.base_layers
                ],
                dim=0,
            )
        else:
            packed_bias = None

        host = self.modules[0]
        host.register_buffer(_PACKED_GATE_UP_WEIGHT, packed_weight, persistent=False)
        host.register_buffer(_PACKED_GATE_UP_BIAS, packed_bias, persistent=False)
        for base in self.base_layers:
            delattr(base, "weight")
            delattr(base, "bias")

        self._state_hook = host.register_state_dict_pre_hook(
            self._reject_packed_state_dict
        )
        for index, module in enumerate(self.modules):
            module.forward = MethodType(self._make_forward(index), module)

    @staticmethod
    def _validate_adapter_state(module, adapter, label):
        if module.disable_adapters or module.merged:
            raise RuntimeError(f"{label}: adapters must be enabled and unmerged")
        if tuple(module.active_adapters) != (adapter,):
            raise RuntimeError(f"{label}: active adapters do not match")
        if (
            tuple(module.lora_A.keys()) != (adapter,)
            or tuple(module.lora_B.keys()) != (adapter,)
        ):
            raise RuntimeError(
                f"{label}: exactly one vanilla LoRA A/B pair is required"
            )
        if adapter in module.lora_variant:
            raise RuntimeError(f"{label}: LoRA variants are not supported")
        if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
            raise RuntimeError(f"{label}: LoRA dropout must be zero")
        if (
            module.lora_A[adapter].bias is not None
            or module.lora_B[adapter].bias is not None
        ):
            raise RuntimeError(f"{label}: LoRA A/B bias is not supported")

    @classmethod
    def validate(cls, modules, label):
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
            raise RuntimeError(f"{label}: only torch.nn.Linear base layers are supported")
        first_weight = first_base.weight
        in_features = first_base.in_features
        out_features = first_base.out_features

        for module in modules:
            if "forward" in module.__dict__:
                raise RuntimeError(
                    f"{label}: module already has an instance forward override"
                )
            if hasattr(module, _PACKED_GATE_UP_WEIGHT) or hasattr(
                module, _PACKED_GATE_UP_BIAS
            ):
                raise RuntimeError(f"{label}: module is already base-packed")
            cls._validate_adapter_state(module, adapter, label)

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
            if weight.dtype != first_weight.dtype or weight.device != first_weight.device:
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
                    or base.bias.dtype != weight.dtype
                    or base.bias.device != weight.device
                ):
                    raise RuntimeError(
                        f"{label}: base bias shape/dtype/device does not match"
                    )
                if base.bias.requires_grad:
                    raise RuntimeError(
                        f"{label}: base projection biases must be frozen"
                    )

    def _reject_packed_state_dict(self, _module, _prefix, _keep_vars):
        raise RuntimeError(
            f"{self.label}: refusing to serialize temporary packed base gate/up "
            "weights; call materialize_standard_modules() first"
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
            group._validate_adapter_state(module, group.adapter, group.label)
            if group._next_module != module_index:
                raise RuntimeError(
                    f"{group.label}: projection call order changed "
                    f"(expected member {group._next_module}, got {module_index})"
                )

            if module_index == 0:
                if group._pending_up is not None:
                    raise RuntimeError(
                        f"{group.label}: incomplete previous packed projection"
                    )
                weight = getattr(group.modules[0], _PACKED_GATE_UP_WEIGHT)
                bias = getattr(group.modules[0], _PACKED_GATE_UP_BIAS)
                projected = torch.nn.functional.linear(x, weight, bias)
                gate_base, group._pending_up = projected.split(
                    group.output_widths,
                    dim=-1,
                )
                group._pending_input = x
                result = gate_base
            else:
                if x is not group._pending_input:
                    raise RuntimeError(
                        f"{group.label}: gate/up did not receive the same input tensor"
                    )
                result = group._pending_up

            result_dtype = result.dtype
            lora_a = module.lora_A[group.adapter]
            lora_b = module.lora_B[group.adapter]
            lora_input = module._cast_input_dtype(x, lora_a.weight.dtype)
            result = result + lora_b(
                module.lora_dropout[group.adapter](lora_a(lora_input))
            ) * module.scaling[group.adapter]

            group._next_module += 1
            if group._next_module == len(group.modules):
                group._pending_input = None
                group._pending_up = None
                group._next_module = 0
            return result.to(result_dtype)

        return packed_forward

    def assert_idle(self):
        if (
            self._next_module
            or self._pending_input is not None
            or self._pending_up is not None
        ):
            raise RuntimeError(
                f"{self.label}: cannot materialize during an incomplete gate/up "
                "forward"
            )

    def materialize_standard_modules(self):
        if self._materialized:
            return
        self.assert_idle()
        host = self.modules[0]
        weight = getattr(host, _PACKED_GATE_UP_WEIGHT)
        bias = getattr(host, _PACKED_GATE_UP_BIAS)
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

        self._state_hook.remove()
        delattr(host, _PACKED_GATE_UP_WEIGHT)
        delattr(host, _PACKED_GATE_UP_BIAS)
        for module, base, restored_weight, restored_bias, has_bias in zip(
            self.modules,
            self.base_layers,
            restored_weights,
            restored_biases,
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
            delattr(module, "forward")
        self._materialized = True


class PackedBaseGateUpPlan:
    """Own packed MLP groups and restore the ordinary module tree."""

    def __init__(self, groups):
        normalised = tuple((tuple(modules), label) for modules, label in groups)
        if not normalised:
            raise RuntimeError("at least one gate/up group is required")

        seen = set()
        for modules, label in normalised:
            _PackedBaseGateUpGroup.validate(modules, label)
            if seen.intersection(map(id, modules)):
                raise RuntimeError(
                    f"{label}: a module belongs to more than one packed group"
                )
            seen.update(map(id, modules))

        installed = []
        try:
            for modules, label in normalised:
                installed.append(_PackedBaseGateUpGroup(modules, label))
        except Exception:
            for group in installed:
                group.materialize_standard_modules()
            raise
        self.groups = tuple(installed)
        self._materialized = False

    def materialize_standard_modules(self):
        if self._materialized:
            return
        for group in self.groups:
            group.assert_idle()
        for group in self.groups:
            group.materialize_standard_modules()
        self._materialized = True


def pack_qwen2_base_gate_up(transformer):
    """Pack frozen gate/up base projections in every Qwen2 decoder layer."""
    try:
        groups = [
            (
                (layer.mlp.gate_proj, layer.mlp.up_proj),
                f"layer {index} base gate/up",
            )
            for index, layer in enumerate(transformer.layers)
        ]
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("unsupported Qwen2 MLP module tree") from exc
    if not groups:
        raise RuntimeError("Qwen2 transformer has no decoder layers")
    return PackedBaseGateUpPlan(groups)




_FUSED_A_PARAMETER = "_packed_a_weight"


class _FusedLoraAGroup:
    """Run several adjacent, dropout-free PEFT LoRA A branches as one GEMM.

    The ordinary ``lora_A.<adapter>.weight`` parameters are temporarily packed into
    one leaf parameter on the first PEFT wrapper. The wrappers' base projections and
    separate LoRA B projections are unchanged. ``materialize_standard_peft`` reverses
    the temporary representation before any state dict can be produced.

    This intentionally supports only the exact vanilla, one-adapter, dropout-free PEFT
    configuration used by this candidate. Unsupported state fails closed rather than
    silently taking a numerically different path.
    """

    def __init__(self, modules, label):
        self.modules = tuple(modules)
        self.label = label
        self.adapter = tuple(self.modules[0].active_adapters)[0]
        self.a_layers = tuple(module.lora_A[self.adapter] for module in self.modules)
        self.ranks = tuple(layer.weight.shape[0] for layer in self.a_layers)
        self._pending_input = None
        self._pending_pieces = None
        self._next_module = 0
        self._materialized = False

        packed = torch.nn.Parameter(
            torch.cat([layer.weight.detach() for layer in self.a_layers], dim=0),
            requires_grad=True,
        )
        host = self.modules[0]
        host.register_parameter(_FUSED_A_PARAMETER, packed)
        for layer in self.a_layers:
            delattr(layer, "weight")

        self._state_hook = host.register_state_dict_pre_hook(self._reject_packed_state_dict)
        for index, module in enumerate(self.modules):
            module.forward = MethodType(self._make_forward(index), module)

    @staticmethod
    def validate(modules, label):
        modules = tuple(modules)
        if len(modules) < 2:
            raise RuntimeError(f"{label}: a fused group needs at least two modules")
        if len({id(module) for module in modules}) != len(modules):
            raise RuntimeError(f"{label}: a module occurs more than once")

        first_adapters = tuple(modules[0].active_adapters)
        if len(first_adapters) != 1:
            raise RuntimeError(f"{label}: exactly one active adapter is required")
        adapter = first_adapters[0]
        first_weight = modules[0].lora_A[adapter].weight
        in_features = first_weight.shape[1]

        for module in modules:
            if "forward" in module.__dict__:
                raise RuntimeError(f"{label}: module already has an instance forward override")
            if hasattr(module, _FUSED_A_PARAMETER):
                raise RuntimeError(f"{label}: module is already fused")
            if module.disable_adapters or module.merged:
                raise RuntimeError(f"{label}: adapters must be enabled and unmerged")
            if tuple(module.active_adapters) != (adapter,):
                raise RuntimeError(f"{label}: active adapters do not match")
            if tuple(module.lora_A.keys()) != (adapter,) or tuple(module.lora_B.keys()) != (adapter,):
                raise RuntimeError(f"{label}: exactly one vanilla LoRA A/B pair is required")
            if adapter in module.lora_variant:
                raise RuntimeError(f"{label}: LoRA variants are not supported")
            if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
                raise RuntimeError(f"{label}: LoRA dropout must be zero")

            a_layer = module.lora_A[adapter]
            b_layer = module.lora_B[adapter]
            weight = a_layer.weight
            if a_layer.bias is not None or b_layer.bias is not None:
                raise RuntimeError(f"{label}: LoRA A/B bias is not supported")
            if weight.ndim != 2 or weight.shape[1] != in_features:
                raise RuntimeError(f"{label}: LoRA A input dimensions do not match")
            if weight.dtype != first_weight.dtype or weight.device != first_weight.device:
                raise RuntimeError(f"{label}: LoRA A dtype/device do not match")
            if not weight.requires_grad:
                raise RuntimeError(f"{label}: LoRA A must be trainable")

    def _reject_packed_state_dict(self, _module, _prefix, _keep_vars):
        raise RuntimeError(
            f"{self.label}: refusing to serialize temporary packed LoRA A weights; "
            "call materialize_standard_peft() first"
        )

    def _make_forward(self, module_index):
        group = self

        def fused_forward(module, x, *args, **kwargs):
            if args or kwargs:
                raise RuntimeError(f"{group.label}: fused LoRA A accepts only the projection input")
            if group._materialized:
                raise RuntimeError(f"{group.label}: stale fused forward after materialization")
            if module.disable_adapters or module.merged:
                raise RuntimeError(f"{group.label}: adapter state changed after fusion")
            if tuple(module.active_adapters) != (group.adapter,):
                raise RuntimeError(f"{group.label}: active adapter changed after fusion")
            if group._next_module != module_index:
                expected = group._next_module
                raise RuntimeError(
                    f"{group.label}: projection call order changed "
                    f"(expected member {expected}, got {module_index})"
                )

            result = module.base_layer(x)
            result_dtype = result.dtype
            if module_index == 0:
                if group._pending_pieces is not None:
                    raise RuntimeError(f"{group.label}: incomplete previous fused projection")
                packed = getattr(group.modules[0], _FUSED_A_PARAMETER)
                fused_input = module._cast_input_dtype(x, packed.dtype)
                projected = torch.nn.functional.linear(fused_input, packed)
                group._pending_input = x
                group._pending_pieces = projected.split(group.ranks, dim=-1)
            elif x is not group._pending_input:
                raise RuntimeError(f"{group.label}: grouped projections did not receive the same tensor")

            piece = group._pending_pieces[module_index]
            branch = module.lora_B[group.adapter](piece) * module.scaling[group.adapter]
            group._next_module += 1
            if group._next_module == len(group.modules):
                group._pending_input = None
                group._pending_pieces = None
                group._next_module = 0
            return (result + branch).to(result_dtype)

        return fused_forward

    def assert_idle(self):
        if self._next_module or self._pending_input is not None or self._pending_pieces is not None:
            raise RuntimeError(f"{self.label}: cannot materialize during an incomplete grouped forward")

    def materialize_standard_peft(self):
        if self._materialized:
            return
        self.assert_idle()
        host = self.modules[0]
        packed = getattr(host, _FUSED_A_PARAMETER)
        chunks = packed.detach().split(self.ranks, dim=0)

        self._state_hook.remove()
        delattr(host, _FUSED_A_PARAMETER)
        for module, layer, chunk in zip(self.modules, self.a_layers, chunks):
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(chunk.contiguous().clone(), requires_grad=packed.requires_grad),
            )
            delattr(module, "forward")
        self._materialized = True


class FusedLoraAPlan:
    """Own all temporary fused groups and restore standard PEFT serialization."""

    def __init__(self, groups):
        normalised = tuple((tuple(modules), label) for modules, label in groups)
        seen = set()
        for modules, label in normalised:
            _FusedLoraAGroup.validate(modules, label)
            overlap = seen.intersection(map(id, modules))
            if overlap:
                raise RuntimeError(f"{label}: a module belongs to more than one fused group")
            seen.update(map(id, modules))

        installed = []
        try:
            for modules, label in normalised:
                installed.append(_FusedLoraAGroup(modules, label))
        except Exception:
            for group in installed:
                group.materialize_standard_peft()
            raise
        self.groups = tuple(installed)
        self._materialized = False

    def materialize_standard_peft(self):
        if self._materialized:
            return
        for group in self.groups:
            group.assert_idle()
        for group in self.groups:
            group.materialize_standard_peft()
        self._materialized = True


def fuse_qwen2_lora_a(transformer):
    """Fuse q/k/v and gate/up LoRA A projections in every Qwen2 decoder layer."""
    groups = []
    for index, layer in enumerate(transformer.layers):
        groups.append(
            ((layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj), f"layer {index} qkv")
        )
        groups.append(((layer.mlp.gate_proj, layer.mlp.up_proj), f"layer {index} gate/up"))
    if not groups:
        raise RuntimeError("Qwen2 transformer has no decoder layers")
    return FusedLoraAPlan(groups)




def fuse_qwen2_lora_a_qkv(transformer):
    """Fuse only q/k/v LoRA-A projections using the reviewed plan."""
    groups = [
        (
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
            f"layer {index} qkv",
        )
        for index, layer in enumerate(transformer.layers)
    ]
    if not groups:
        raise RuntimeError("Qwen2 transformer has no decoder layers")
    return FusedLoraAPlan(groups)


def fuse_qwen2_lora_a_gate_up(transformer):
    """Fuse only gate/up LoRA-A projections using the reviewed plan."""
    groups = [
        (
            (layer.mlp.gate_proj, layer.mlp.up_proj),
            f"layer {index} gate/up",
        )
        for index, layer in enumerate(transformer.layers)
    ]
    if not groups:
        raise RuntimeError("Qwen2 transformer has no decoder layers")
    return FusedLoraAPlan(groups)
