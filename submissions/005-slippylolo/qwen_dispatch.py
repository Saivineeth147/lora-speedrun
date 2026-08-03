"""Parent-level packed dispatch for the pinned Qwen2.5 LoRA runtime.

The ordinary Qwen2 attention and MLP implementations dispatch five adjacent
LoRA projections separately. For the exact benchmark architecture, this module
row-packs each attention parent's q/k/v frozen weights and each MLP parent's
gate/up frozen weights. Wrapper-backed variants also row-pack their LoRA-A
weights; the direct-packed variant consumes LoRA-A tensors constructed in that
form from birth. Each parent then executes one frozen-base ``F.linear`` and one
LoRA-A ``F.linear`` before splitting the independent LoRA-B branches.

This is deliberately not a generic Transformers optimization.  Installation
authenticates the pinned Transformers implementation and the small PEFT-like
wrapper contract used by both PEFT 0.19.1 and ``StaticLoraLinear``, or accepts
the separately authenticated wrapper-free direct-packed descriptor. Unsupported
structure fails before the model is mutated. ``materialize_standard_peft``
restores the original parent forwards and independent frozen Linear storage. In
wrapper mode it also restores the original LoRA-A Parameter objects; in direct
mode the packed adapter remains registered for the direct standard-artifact
writer.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from importlib import import_module, metadata
from numbers import Real
from types import MethodType

import torch


ADAPTER_NAME = "default"
EXPECTED_QWEN2_LAYERS = 28
PINNED_TRANSFORMERS_VERSION = "5.14.1"
_LINEAR = torch.nn.functional.linear
_SILU = torch.nn.functional.silu

_PACKED_BASE_WEIGHT = "_packed_parent_packed_base_weight"
_PACKED_BASE_BIAS = "_packed_parent_packed_base_bias"
_PACKED_LORA_A_WEIGHT = "_packed_parent_packed_lora_a_weight"

_LOCAL_CALL_HOOKS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
)
_GLOBAL_CALL_HOOKS = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
    "_global_forward_pre_hooks_with_kwargs",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_hooks_always_called",
)


@dataclass(frozen=True)
class _PinnedQwen2:
    model_type: type
    decoder_layer_type: type
    attention_type: type
    mlp_type: type
    apply_rotary_pos_emb: object
    attention_interfaces: object
    eager_attention_forward: object


@dataclass(frozen=True)
class _Projection:
    name: str
    wrapper: torch.nn.Module
    base: torch.nn.Linear
    lora_a: torch.nn.Linear
    lora_b: torch.nn.Linear
    dropout: torch.nn.Identity
    scaling: Real
    cast_input_dtype_enabled: bool

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def rank(self) -> int:
        return self.lora_a.out_features


@dataclass(frozen=True)
class _GroupSpec:
    parent: torch.nn.Module
    projections: tuple[_Projection, ...]
    label: str
    kind: str
    pinned: _PinnedQwen2
    output_projection: _Projection | None = None
    down_projection: torch.nn.Linear | None = None
    attention_interface: object | None = None
    direct_adapter: object | None = None
    direct_spec: object | None = None
    layer_index: int | None = None


def _signature_shape(function) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    """Return the annotation-independent part of a callable signature."""

    parameters = inspect.signature(function).parameters.values()
    empty = inspect.Parameter.empty
    return tuple(
        (
            parameter.name,
            parameter.kind,
            empty if parameter.default is empty else parameter.default,
        )
        for parameter in parameters
    )


def _require_signature(function, expected, label: str) -> None:
    try:
        observed = _signature_shape(function)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}: callable signature is unavailable") from exc
    if observed != expected:
        raise RuntimeError(
            f"{label}: callable signature differs from the pinned contract; "
            f"got {inspect.signature(function)}"
        )


_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
_EMPTY = inspect.Parameter.empty
_ATTENTION_SIGNATURE = (
    ("self", _POSITIONAL, _EMPTY),
    ("hidden_states", _POSITIONAL, _EMPTY),
    ("position_embeddings", _POSITIONAL, _EMPTY),
    ("attention_mask", _POSITIONAL, _EMPTY),
    ("past_key_values", _POSITIONAL, None),
    ("kwargs", _VAR_KEYWORD, _EMPTY),
)
_MLP_SIGNATURE = (
    ("self", _POSITIONAL, _EMPTY),
    ("x", _POSITIONAL, _EMPTY),
)
_WRAPPER_SIGNATURE = (
    ("self", _POSITIONAL, _EMPTY),
    ("x", _POSITIONAL, _EMPTY),
    ("args", _VAR_POSITIONAL, _EMPTY),
    ("kwargs", _VAR_KEYWORD, _EMPTY),
)
_CAST_SIGNATURE = (
    ("x", _POSITIONAL, _EMPTY),
    ("dtype", _POSITIONAL, _EMPTY),
)


def _load_pinned_qwen2() -> _PinnedQwen2:
    try:
        installed_version = metadata.version("transformers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Transformers is unavailable") from exc
    if installed_version != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "parent dispatch requires Transformers "
            f"{PINNED_TRANSFORMERS_VERSION}, got {installed_version}"
        )

    module = import_module("transformers.models.qwen2.modeling_qwen2")
    required = (
        "Qwen2Model",
        "Qwen2DecoderLayer",
        "Qwen2Attention",
        "Qwen2MLP",
        "apply_rotary_pos_emb",
        "ALL_ATTENTION_FUNCTIONS",
        "eager_attention_forward",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(
            "pinned Qwen2 implementation is missing: " + ", ".join(missing)
        )

    _require_signature(
        module.Qwen2Attention.forward,
        _ATTENTION_SIGNATURE,
        "Qwen2Attention.forward",
    )
    _require_signature(
        module.Qwen2MLP.forward,
        _MLP_SIGNATURE,
        "Qwen2MLP.forward",
    )
    attention_globals = module.Qwen2Attention.forward.__globals__
    expected_globals = (
        ("apply_rotary_pos_emb", module.apply_rotary_pos_emb),
        ("ALL_ATTENTION_FUNCTIONS", module.ALL_ATTENTION_FUNCTIONS),
        ("eager_attention_forward", module.eager_attention_forward),
    )
    for name, expected in expected_globals:
        if attention_globals.get(name) is not expected:
            raise RuntimeError(
                f"Qwen2Attention.forward global {name} differs from the "
                "pinned implementation"
            )
    if not callable(module.apply_rotary_pos_emb):
        raise RuntimeError("pinned Qwen2 rotary function is not callable")
    if not callable(module.eager_attention_forward):
        raise RuntimeError("pinned Qwen2 eager attention function is not callable")
    if not callable(
        getattr(module.ALL_ATTENTION_FUNCTIONS, "get_interface", None)
    ):
        raise RuntimeError("pinned attention registry has no get_interface")

    return _PinnedQwen2(
        model_type=module.Qwen2Model,
        decoder_layer_type=module.Qwen2DecoderLayer,
        attention_type=module.Qwen2Attention,
        mlp_type=module.Qwen2MLP,
        apply_rotary_pos_emb=module.apply_rotary_pos_emb,
        attention_interfaces=module.ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward=module.eager_attention_forward,
    )


def _require_no_global_module_hooks() -> None:
    module_impl = torch.nn.modules.module
    active = tuple(
        name
        for name in _GLOBAL_CALL_HOOKS
        if getattr(module_impl, name, None)
    )
    if active:
        raise RuntimeError(
            "parent dispatch cannot preserve global nn.Module hooks: "
            + ", ".join(active)
        )


def _require_no_local_call_hooks(module: torch.nn.Module, label: str) -> None:
    active = tuple(
        name for name in _LOCAL_CALL_HOOKS if getattr(module, name, None)
    )
    if active:
        raise RuntimeError(
            f"{label}: parent dispatch would bypass registered module hooks: "
            + ", ".join(active)
        )


def _mapping_member(mapping, label: str):
    try:
        keys = tuple(mapping.keys())
    except AttributeError as exc:
        raise RuntimeError(f"{label}: adapter collection is not mapping-like") from exc
    if keys != (ADAPTER_NAME,):
        raise RuntimeError(
            f"{label}: exactly the {ADAPTER_NAME!r} adapter is required"
        )
    try:
        return mapping[ADAPTER_NAME]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"{label}: default adapter member is unavailable") from exc


def _validate_projection(wrapper, name: str) -> _Projection:
    if not isinstance(wrapper, torch.nn.Module):
        raise RuntimeError(f"{name}: LoRA wrapper must be an nn.Module")
    if "forward" in wrapper.__dict__:
        raise RuntimeError(f"{name}: wrapper already has an instance forward")
    _require_signature(type(wrapper).forward, _WRAPPER_SIGNATURE, f"{name}.forward")
    _require_no_local_call_hooks(wrapper, name)

    try:
        active_adapters = tuple(wrapper.active_adapters)
        disable_adapters = wrapper.disable_adapters
        merged = wrapper.merged
        lora_variant = wrapper.lora_variant
        cast_enabled = wrapper.cast_input_dtype_enabled
        caster = wrapper._cast_input_dtype
    except AttributeError as exc:
        raise RuntimeError(f"{name}: incomplete PEFT-like wrapper contract") from exc
    if active_adapters != (ADAPTER_NAME,):
        raise RuntimeError(f"{name}: only the default adapter may be active")
    if disable_adapters is not False or merged is not False:
        raise RuntimeError(f"{name}: adapter must be enabled and unmerged")
    if lora_variant != {}:
        raise RuntimeError(f"{name}: LoRA variants are unsupported")
    if type(cast_enabled) is not bool:
        raise RuntimeError(f"{name}: input-cast state must be boolean")
    if not callable(caster):
        raise RuntimeError(f"{name}: input-cast helper is unavailable")
    _require_signature(caster, _CAST_SIGNATURE, f"{name}._cast_input_dtype")

    try:
        base = wrapper.base_layer
        lora_a = _mapping_member(wrapper.lora_A, f"{name}.lora_A")
        lora_b = _mapping_member(wrapper.lora_B, f"{name}.lora_B")
        dropout = _mapping_member(
            wrapper.lora_dropout,
            f"{name}.lora_dropout",
        )
        scaling = _mapping_member(wrapper.scaling, f"{name}.scaling")
    except AttributeError as exc:
        raise RuntimeError(f"{name}: incomplete LoRA projection state") from exc

    if type(base) is not torch.nn.Linear:
        raise RuntimeError(f"{name}: base layer must be exact nn.Linear")
    if type(lora_a) is not torch.nn.Linear or type(lora_b) is not torch.nn.Linear:
        raise RuntimeError(f"{name}: LoRA A/B must be exact nn.Linear")
    if type(dropout) is not torch.nn.Identity:
        raise RuntimeError(f"{name}: LoRA dropout must be exact Identity")
    for bypassed, suffix in (
        (base, "base_layer"),
        (lora_a, "lora_A.default"),
        (lora_b, "lora_B.default"),
        (dropout, "lora_dropout.default"),
    ):
        _require_no_local_call_hooks(bypassed, f"{name}.{suffix}")

    if isinstance(scaling, bool) or not isinstance(scaling, Real):
        raise RuntimeError(f"{name}: LoRA scaling must be a real scalar")
    if not math.isfinite(float(scaling)):
        raise RuntimeError(f"{name}: LoRA scaling must be finite")
    if (
        base.weight.ndim != 2
        or base.weight.shape != (base.out_features, base.in_features)
        or not base.weight.is_contiguous()
    ):
        raise RuntimeError(f"{name}: base weight dimensions are inconsistent")
    if base.weight.requires_grad:
        raise RuntimeError(f"{name}: base weight must be frozen")
    if base.weight.grad is not None:
        raise RuntimeError(f"{name}: base weight already has a gradient")
    if base.bias is not None:
        if base.bias.shape != (base.out_features,):
            raise RuntimeError(f"{name}: base bias dimensions are inconsistent")
        if (
            not base.bias.is_contiguous()
            or base.bias.dtype != base.weight.dtype
            or base.bias.device != base.weight.device
            or base.bias.requires_grad
            or base.bias.grad is not None
        ):
            raise RuntimeError(f"{name}: base bias state is unsupported")

    a_weight = lora_a.weight
    b_weight = lora_b.weight
    if lora_a.bias is not None or lora_b.bias is not None:
        raise RuntimeError(f"{name}: LoRA A/B bias is unsupported")
    if (
        a_weight.ndim != 2
        or a_weight.shape != (lora_a.out_features, base.in_features)
        or b_weight.ndim != 2
        or b_weight.shape != (base.out_features, lora_a.out_features)
        or not a_weight.is_contiguous()
        or not b_weight.is_contiguous()
    ):
        raise RuntimeError(f"{name}: LoRA A/B dimensions are inconsistent")
    if (
        a_weight.dtype != b_weight.dtype
        or a_weight.device != base.weight.device
        or b_weight.device != base.weight.device
    ):
        raise RuntimeError(f"{name}: LoRA A/B dtype or device is unsupported")
    if not a_weight.requires_grad or not b_weight.requires_grad:
        raise RuntimeError(f"{name}: LoRA A/B must be trainable")
    if a_weight.grad is not None or b_weight.grad is not None:
        raise RuntimeError(f"{name}: LoRA A/B already has a gradient")

    return _Projection(
        name=name,
        wrapper=wrapper,
        base=base,
        lora_a=lora_a,
        lora_b=lora_b,
        dropout=dropout,
        scaling=scaling,
        cast_input_dtype_enabled=cast_enabled,
    )


def _validate_group(spec: _GroupSpec, *, biases: bool) -> None:
    projections = spec.projections
    expected_members = 3 if spec.kind == "attention" else 2
    if len(projections) != expected_members:
        raise RuntimeError(
            f"{spec.label}: expected {expected_members} packed projections"
        )
    if len({id(item.wrapper) for item in projections}) != expected_members:
        raise RuntimeError(f"{spec.label}: projection wrappers must be distinct")

    first = projections[0]
    first_cast_function = getattr(
        first.wrapper._cast_input_dtype,
        "__func__",
        first.wrapper._cast_input_dtype,
    )
    for projection in projections:
        if projection.in_features != first.in_features:
            raise RuntimeError(f"{spec.label}: projection input widths differ")
        if (
            projection.base.weight.dtype != first.base.weight.dtype
            or projection.base.weight.device != first.base.weight.device
        ):
            raise RuntimeError(
                f"{spec.label}: base projection dtype/device differ"
            )
        if (
            projection.lora_a.weight.dtype != first.lora_a.weight.dtype
            or projection.lora_a.weight.device != first.lora_a.weight.device
            or projection.rank != first.rank
        ):
            raise RuntimeError(
                f"{spec.label}: LoRA-A dtype/device/rank differ"
            )
        if projection.cast_input_dtype_enabled != first.cast_input_dtype_enabled:
            raise RuntimeError(f"{spec.label}: input-cast states differ")
        cast_function = getattr(
            projection.wrapper._cast_input_dtype,
            "__func__",
            projection.wrapper._cast_input_dtype,
        )
        if cast_function is not first_cast_function:
            raise RuntimeError(
                f"{spec.label}: input-cast implementations differ"
            )
        if (projection.base.bias is not None) is not biases:
            expected = "present" if biases else "absent"
            raise RuntimeError(
                f"{projection.name}: pinned base bias must be {expected}"
            )


def _validate_attention(
    *,
    attention,
    config,
    layer_index: int,
    pinned: _PinnedQwen2,
) -> _GroupSpec:
    label = f"layer {layer_index} attention parent dispatch"
    if type(attention) is not pinned.attention_type:
        raise RuntimeError(f"{label}: parent is not exact Qwen2Attention")
    if "forward" in attention.__dict__:
        raise RuntimeError(f"{label}: parent already has an instance forward")
    if attention.config is not config:
        raise RuntimeError(f"{label}: config identity differs")
    if attention.layer_idx != layer_index:
        raise RuntimeError(f"{label}: layer index differs")
    if config._attn_implementation != "sdpa":
        raise RuntimeError(f"{label}: only the pinned SDPA path is supported")
    if attention.attention_dropout != 0.0:
        raise RuntimeError(f"{label}: attention dropout must be zero")
    if attention.sliding_window is not None:
        raise RuntimeError(f"{label}: sliding-window attention is unsupported")

    expected_head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )
    if (
        type(attention.head_dim) is not int
        or attention.head_dim != expected_head_dim
        or attention.head_dim < 1
    ):
        raise RuntimeError(f"{label}: head dimension differs")
    if attention.num_key_value_groups != (
        config.num_attention_heads // config.num_key_value_heads
    ):
        raise RuntimeError(f"{label}: key/value group count differs")
    if attention.scaling != attention.head_dim**-0.5:
        raise RuntimeError(f"{label}: attention scaling differs")
    if not isinstance(attention.o_proj, torch.nn.Module):
        raise RuntimeError(f"{label}: output projection is unavailable")

    q = _validate_projection(attention.q_proj, f"layer {layer_index} q_proj")
    k = _validate_projection(attention.k_proj, f"layer {layer_index} k_proj")
    v = _validate_projection(attention.v_proj, f"layer {layer_index} v_proj")
    output = _validate_projection(
        attention.o_proj,
        f"layer {layer_index} o_proj",
    )
    hidden_size = config.hidden_size
    q_width = config.num_attention_heads * attention.head_dim
    kv_width = config.num_key_value_heads * attention.head_dim
    if (
        q.in_features != hidden_size
        or k.in_features != hidden_size
        or v.in_features != hidden_size
        or (q.out_features, k.out_features, v.out_features)
        != (q_width, kv_width, kv_width)
        or output.in_features != q_width
        or output.out_features != hidden_size
    ):
        raise RuntimeError(f"{label}: pinned projection dimensions differ")
    if (
        output.base.weight.dtype != q.base.weight.dtype
        or output.base.weight.device != q.base.weight.device
    ):
        raise RuntimeError(f"{label}: output projection dtype/device differs")
    attention_interface = pinned.attention_interfaces.get_interface(
        config._attn_implementation,
        pinned.eager_attention_forward,
    )
    if not callable(attention_interface):
        raise RuntimeError(f"{label}: resolved attention interface is not callable")

    group = _GroupSpec(
        parent=attention,
        projections=(q, k, v),
        label=label,
        kind="attention",
        pinned=pinned,
        output_projection=output,
        attention_interface=attention_interface,
    )
    _validate_group(group, biases=True)
    return group


def _validate_mlp(
    *,
    mlp,
    config,
    layer_index: int,
    pinned: _PinnedQwen2,
) -> _GroupSpec:
    label = f"layer {layer_index} MLP parent dispatch"
    if type(mlp) is not pinned.mlp_type:
        raise RuntimeError(f"{label}: parent is not exact Qwen2MLP")
    if "forward" in mlp.__dict__:
        raise RuntimeError(f"{label}: parent already has an instance forward")
    if mlp.config is not config:
        raise RuntimeError(f"{label}: config identity differs")
    if (
        mlp.hidden_size != config.hidden_size
        or mlp.intermediate_size != config.intermediate_size
    ):
        raise RuntimeError(f"{label}: pinned MLP dimensions differ")
    if config.hidden_act != "silu" or not callable(mlp.act_fn):
        raise RuntimeError(f"{label}: pinned SiLU activation differs")
    _require_no_local_call_hooks(mlp.act_fn, f"{label} activation")
    if type(mlp.down_proj) is not torch.nn.Linear:
        raise RuntimeError(f"{label}: down projection must be exact nn.Linear")
    if (
        mlp.down_proj.in_features != config.intermediate_size
        or mlp.down_proj.out_features != config.hidden_size
        or mlp.down_proj.bias is not None
        or mlp.down_proj.weight.requires_grad
        or mlp.down_proj.weight.grad is not None
    ):
        raise RuntimeError(f"{label}: down projection state differs")
    _require_no_local_call_hooks(mlp.down_proj, f"{label} down_proj")

    gate = _validate_projection(
        mlp.gate_proj,
        f"layer {layer_index} gate_proj",
    )
    up = _validate_projection(mlp.up_proj, f"layer {layer_index} up_proj")
    if (
        gate.in_features != config.hidden_size
        or up.in_features != config.hidden_size
        or gate.out_features != config.intermediate_size
        or up.out_features != config.intermediate_size
    ):
        raise RuntimeError(f"{label}: pinned projection dimensions differ")

    group = _GroupSpec(
        parent=mlp,
        projections=(gate, up),
        label=label,
        kind="mlp",
        pinned=pinned,
        down_projection=mlp.down_proj,
    )
    _validate_group(group, biases=False)
    if (
        mlp.down_proj.weight.dtype != gate.base.weight.dtype
        or mlp.down_proj.weight.device != gate.base.weight.device
    ):
        raise RuntimeError(f"{label}: down projection dtype/device differs")
    return group


def _validate_direct_base(
    base,
    *,
    name: str,
    in_features: int,
    out_features: int,
    bias: bool,
) -> None:
    if type(base) is not torch.nn.Linear:
        raise RuntimeError(f"{name}: direct base must be exact nn.Linear")
    _require_no_local_call_hooks(base, name)
    if (
        base.in_features != in_features
        or base.out_features != out_features
        or base.weight.shape != (out_features, in_features)
        or not base.weight.is_contiguous()
        or base.weight.requires_grad
        or base.weight.grad is not None
        or (base.bias is not None) is not bias
    ):
        raise RuntimeError(f"{name}: direct frozen base state differs")
    if base.bias is not None and (
        base.bias.shape != (out_features,)
        or not base.bias.is_contiguous()
        or base.bias.dtype != base.weight.dtype
        or base.bias.device != base.weight.device
        or base.bias.requires_grad
        or base.bias.grad is not None
    ):
        raise RuntimeError(f"{name}: direct frozen base bias differs")


def _validate_direct_parameter(
    parameter,
    *,
    name: str,
    shape: tuple[int, ...],
    reference: torch.Tensor,
) -> None:
    if type(parameter) is not torch.nn.Parameter:
        raise RuntimeError(f"{name}: direct adapter must be an exact Parameter")
    if (
        parameter.shape != shape
        or not parameter.is_contiguous()
        or parameter.dtype != reference.dtype
        or parameter.device != reference.device
        or not parameter.requires_grad
        or parameter.grad is not None
    ):
        raise RuntimeError(f"{name}: direct adapter tensor state differs")


def _validate_direct_scalings(values, *, count: int, label: str) -> tuple[Real, ...]:
    try:
        scalings = tuple(values)
    except TypeError as exc:
        raise RuntimeError(f"{label}: direct scalings are not iterable") from exc
    if len(scalings) != count or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in scalings
    ):
        raise RuntimeError(f"{label}: direct LoRA scalings differ")
    return scalings


def _require_direct_registration(parent, parameters, label: str) -> None:
    registered = tuple(parent._parameters.values())
    for parameter in parameters:
        if not any(candidate is parameter for candidate in registered):
            raise RuntimeError(
                f"{label}: direct adapter is not registered on its parent"
            )


def _validate_direct_attention(
    *,
    attention,
    config,
    layer_index: int,
    pinned: _PinnedQwen2,
    direct_adapter,
    descriptor,
) -> _GroupSpec:
    label = f"layer {layer_index} direct attention parent dispatch"
    if type(attention) is not pinned.attention_type:
        raise RuntimeError(f"{label}: parent is not exact Qwen2Attention")
    if "forward" in attention.__dict__:
        raise RuntimeError(f"{label}: parent already has an instance forward")
    if attention.config is not config or attention.layer_idx != layer_index:
        raise RuntimeError(f"{label}: parent configuration differs")
    if (
        config._attn_implementation != "sdpa"
        or attention.attention_dropout != 0.0
        or attention.sliding_window is not None
    ):
        raise RuntimeError(f"{label}: pinned attention mode differs")
    expected_head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )
    if (
        type(attention.head_dim) is not int
        or attention.head_dim != expected_head_dim
        or attention.head_dim < 1
        or attention.num_key_value_groups
        != config.num_attention_heads // config.num_key_value_heads
        or attention.scaling != attention.head_dim**-0.5
    ):
        raise RuntimeError(f"{label}: pinned head geometry differs")

    try:
        kind = descriptor.kind
        base_layers = tuple(descriptor.base_layers)
        packed_a = descriptor.packed_lora_a
        b_weights = tuple(descriptor.lora_b_weights)
        scalings = _validate_direct_scalings(
            descriptor.scalings,
            count=3,
            label=label,
        )
        output_base = descriptor.output_base
        output_a = descriptor.output_lora_a
        output_b = descriptor.output_lora_b
        output_scaling = descriptor.output_scaling
    except AttributeError as exc:
        raise RuntimeError(f"{label}: direct descriptor is incomplete") from exc
    if kind != "attention" or len(base_layers) != 3 or len(b_weights) != 3:
        raise RuntimeError(f"{label}: direct descriptor inventory differs")
    if base_layers != (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
    ) or output_base is not attention.o_proj:
        raise RuntimeError(f"{label}: direct base identity differs")

    hidden_size = config.hidden_size
    q_width = config.num_attention_heads * attention.head_dim
    kv_width = config.num_key_value_heads * attention.head_dim
    widths = (q_width, kv_width, kv_width)
    for name, base, width in zip(
        ("q_proj", "k_proj", "v_proj"),
        base_layers,
        widths,
    ):
        _validate_direct_base(
            base,
            name=f"layer {layer_index} {name}",
            in_features=hidden_size,
            out_features=width,
            bias=True,
        )
    _validate_direct_base(
        output_base,
        name=f"layer {layer_index} o_proj",
        in_features=q_width,
        out_features=hidden_size,
        bias=False,
    )
    first_base = base_layers[0].weight
    if any(
        base.weight.dtype != first_base.dtype
        or base.weight.device != first_base.device
        for base in (*base_layers, output_base)
    ):
        raise RuntimeError(f"{label}: direct base dtype/device differ")

    ranks = tuple(weight.shape[1] if weight.ndim == 2 else -1 for weight in b_weights)
    if len(set(ranks)) != 1 or ranks[0] < 1:
        raise RuntimeError(f"{label}: direct q/k/v ranks differ")
    for name, weight, width, rank in zip(
        ("q", "k", "v"),
        b_weights,
        widths,
        ranks,
    ):
        _validate_direct_parameter(
            weight,
            name=f"layer {layer_index} direct {name} LoRA-B",
            shape=(width, rank),
            reference=first_base,
        )
    _validate_direct_parameter(
        packed_a,
        name=f"layer {layer_index} direct packed qkv LoRA-A",
        shape=(sum(ranks), hidden_size),
        reference=first_base,
    )
    output_rank = output_a.shape[0] if getattr(output_a, "ndim", 0) == 2 else -1
    if output_rank < 1:
        raise RuntimeError(f"{label}: direct output rank differs")
    _validate_direct_parameter(
        output_a,
        name=f"layer {layer_index} direct o LoRA-A",
        shape=(output_rank, q_width),
        reference=first_base,
    )
    _validate_direct_parameter(
        output_b,
        name=f"layer {layer_index} direct o LoRA-B",
        shape=(hidden_size, output_rank),
        reference=first_base,
    )
    output_scaling = _validate_direct_scalings(
        (output_scaling,),
        count=1,
        label=f"{label} output",
    )[0]
    _require_direct_registration(
        attention,
        (packed_a, *b_weights, output_a, output_b),
        label,
    )

    attention_interface = pinned.attention_interfaces.get_interface(
        config._attn_implementation,
        pinned.eager_attention_forward,
    )
    if not callable(attention_interface):
        raise RuntimeError(f"{label}: resolved attention interface is not callable")
    return _GroupSpec(
        parent=attention,
        projections=(),
        label=label,
        kind="attention",
        pinned=pinned,
        attention_interface=attention_interface,
        direct_adapter=direct_adapter,
        direct_spec=descriptor,
        layer_index=layer_index,
    )


def _validate_direct_mlp(
    *,
    mlp,
    config,
    layer_index: int,
    pinned: _PinnedQwen2,
    direct_adapter,
    descriptor,
) -> _GroupSpec:
    label = f"layer {layer_index} direct MLP parent dispatch"
    if type(mlp) is not pinned.mlp_type:
        raise RuntimeError(f"{label}: parent is not exact Qwen2MLP")
    if "forward" in mlp.__dict__:
        raise RuntimeError(f"{label}: parent already has an instance forward")
    if (
        mlp.config is not config
        or mlp.hidden_size != config.hidden_size
        or mlp.intermediate_size != config.intermediate_size
        or config.hidden_act != "silu"
        or not callable(mlp.act_fn)
    ):
        raise RuntimeError(f"{label}: pinned MLP configuration differs")
    _require_no_local_call_hooks(mlp.act_fn, f"{label} activation")

    try:
        kind = descriptor.kind
        base_layers = tuple(descriptor.base_layers)
        packed_a = descriptor.packed_lora_a
        b_weights = tuple(descriptor.lora_b_weights)
        scalings = _validate_direct_scalings(
            descriptor.scalings,
            count=2,
            label=label,
        )
    except AttributeError as exc:
        raise RuntimeError(f"{label}: direct descriptor is incomplete") from exc
    if kind != "mlp" or len(base_layers) != 2 or len(b_weights) != 2:
        raise RuntimeError(f"{label}: direct descriptor inventory differs")
    if base_layers != (mlp.gate_proj, mlp.up_proj):
        raise RuntimeError(f"{label}: direct base identity differs")
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    for name, base in zip(("gate_proj", "up_proj"), base_layers):
        _validate_direct_base(
            base,
            name=f"layer {layer_index} {name}",
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        )
    _validate_direct_base(
        mlp.down_proj,
        name=f"layer {layer_index} down_proj",
        in_features=intermediate_size,
        out_features=hidden_size,
        bias=False,
    )
    first_base = base_layers[0].weight
    if any(
        base.weight.dtype != first_base.dtype
        or base.weight.device != first_base.device
        for base in (*base_layers, mlp.down_proj)
    ):
        raise RuntimeError(f"{label}: direct base dtype/device differ")
    ranks = tuple(weight.shape[1] if weight.ndim == 2 else -1 for weight in b_weights)
    if len(set(ranks)) != 1 or ranks[0] < 1:
        raise RuntimeError(f"{label}: direct gate/up ranks differ")
    for name, weight, rank in zip(("gate", "up"), b_weights, ranks):
        _validate_direct_parameter(
            weight,
            name=f"layer {layer_index} direct {name} LoRA-B",
            shape=(intermediate_size, rank),
            reference=first_base,
        )
    _validate_direct_parameter(
        packed_a,
        name=f"layer {layer_index} direct packed gate/up LoRA-A",
        shape=(sum(ranks), hidden_size),
        reference=first_base,
    )
    _require_direct_registration(mlp, (packed_a, *b_weights), label)
    return _GroupSpec(
        parent=mlp,
        projections=(),
        label=label,
        kind="mlp",
        pinned=pinned,
        down_projection=mlp.down_proj,
        direct_adapter=direct_adapter,
        direct_spec=descriptor,
        layer_index=layer_index,
    )


def _validate_transformer(
    transformer,
    direct_adapter=None,
) -> tuple[_GroupSpec, ...]:
    pinned = _load_pinned_qwen2()
    _require_no_global_module_hooks()
    if type(transformer) is not pinned.model_type:
        raise RuntimeError("parent dispatch requires exact Qwen2Model")
    if type(transformer.layers) is not torch.nn.ModuleList:
        raise RuntimeError("pinned Qwen2 layers must be an exact ModuleList")
    layers = tuple(transformer.layers)
    if len(layers) != EXPECTED_QWEN2_LAYERS:
        raise RuntimeError(
            "parent dispatch requires exactly "
            f"{EXPECTED_QWEN2_LAYERS} decoder layers; got {len(layers)}"
        )
    config = transformer.config
    if config.model_type != "qwen2" or config.num_hidden_layers != len(layers):
        raise RuntimeError("pinned Qwen2 model configuration differs")
    if direct_adapter is not None:
        if getattr(direct_adapter, "transformer", None) is not transformer:
            raise RuntimeError(
                "direct packed adapter belongs to a different transformer"
            )
        if not callable(
            getattr(direct_adapter, "parent_dispatch_spec", None)
        ):
            raise RuntimeError(
                "direct packed adapter has no parent_dispatch_spec method"
            )

    groups = []
    seen_wrappers = set()
    direct_parameters = []
    for layer_index, layer in enumerate(layers):
        if type(layer) is not pinned.decoder_layer_type:
            raise RuntimeError(
                f"layer {layer_index}: decoder is not exact Qwen2DecoderLayer"
            )
        if direct_adapter is None:
            attention_group = _validate_attention(
                attention=layer.self_attn,
                config=config,
                layer_index=layer_index,
                pinned=pinned,
            )
            mlp_group = _validate_mlp(
                mlp=layer.mlp,
                config=config,
                layer_index=layer_index,
                pinned=pinned,
            )
            for projection in (
                *attention_group.projections,
                attention_group.output_projection,
                *mlp_group.projections,
            ):
                if projection is None:  # pragma: no cover
                    raise RuntimeError(
                        "internal parent dispatch inventory is incomplete"
                    )
                identity = id(projection.wrapper)
                if identity in seen_wrappers:
                    raise RuntimeError(
                        f"{projection.name}: a LoRA wrapper appears more than once"
                    )
                seen_wrappers.add(identity)
        else:
            try:
                attention_descriptor = direct_adapter.parent_dispatch_spec(
                    layer_index,
                    "attention",
                )
                mlp_descriptor = direct_adapter.parent_dispatch_spec(
                    layer_index,
                    "mlp",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"layer {layer_index}: direct packed descriptor failed"
                ) from exc
            attention_group = _validate_direct_attention(
                attention=layer.self_attn,
                config=config,
                layer_index=layer_index,
                pinned=pinned,
                direct_adapter=direct_adapter,
                descriptor=attention_descriptor,
            )
            mlp_group = _validate_direct_mlp(
                mlp=layer.mlp,
                config=config,
                layer_index=layer_index,
                pinned=pinned,
                direct_adapter=direct_adapter,
                descriptor=mlp_descriptor,
            )
            direct_parameters.extend(
                (
                    attention_descriptor.packed_lora_a,
                    *attention_descriptor.lora_b_weights,
                    attention_descriptor.output_lora_a,
                    attention_descriptor.output_lora_b,
                    mlp_descriptor.packed_lora_a,
                    *mlp_descriptor.lora_b_weights,
                )
            )
        groups.extend((attention_group, mlp_group))
    if direct_adapter is not None:
        if len({id(parameter) for parameter in direct_parameters}) != len(
            direct_parameters
        ):
            raise RuntimeError("a direct packed adapter Parameter appears twice")
        observed = tuple(
            parameter
            for parameter in transformer.parameters()
            if parameter.requires_grad
        )
        if len(observed) != len(direct_parameters) or {
            id(parameter) for parameter in observed
        } != {id(parameter) for parameter in direct_parameters}:
            raise RuntimeError(
                "direct packed trainable traversal differs from its descriptors"
            )
    return tuple(groups)


class _PackedParent:
    """Own one parent's packed storage and validation-free hot continuation."""

    def __init__(self, spec: _GroupSpec) -> None:
        self.spec = spec
        self.parent = spec.parent
        self.direct = spec.direct_spec is not None
        self.projections = spec.projections
        if self.direct:
            descriptor = spec.direct_spec
            self.base_layers = tuple(descriptor.base_layers)
            self.a_layers = ()
            self.b_layers = ()
            self.a_weights = ()
            self.b_weights = tuple(descriptor.lora_b_weights)
            self.scalings = tuple(descriptor.scalings)
            self.output_widths = tuple(
                base.out_features for base in self.base_layers
            )
            self.ranks = tuple(weight.shape[1] for weight in self.b_weights)
        else:
            self.base_layers = tuple(item.base for item in self.projections)
            self.a_layers = tuple(item.lora_a for item in self.projections)
            self.b_layers = tuple(item.lora_b for item in self.projections)
            self.a_weights = tuple(layer.weight for layer in self.a_layers)
            self.b_weights = tuple(layer.weight for layer in self.b_layers)
            self.scalings = tuple(item.scaling for item in self.projections)
            self.output_widths = tuple(
                item.out_features for item in self.projections
            )
            self.ranks = tuple(item.rank for item in self.projections)
        self.base_weights = tuple(layer.weight for layer in self.base_layers)
        self.base_biases = tuple(layer.bias for layer in self.base_layers)
        self.input_features = self.base_layers[0].in_features
        self.base_dtype = self.base_weights[0].dtype
        self.adapter_dtype = (
            spec.direct_spec.packed_lora_a.dtype
            if self.direct
            else self.a_weights[0].dtype
        )
        self.cast_adapter_input = (
            True
            if self.direct
            else self.projections[0].cast_input_dtype_enabled
        )
        self.device = self.base_weights[0].device
        self._state_hook = None
        self._installed_forward = None
        self._materialized = False

        # Cache the complete parent continuation once.  The installed forwards
        # below intentionally contain no wrapper/mapping lookup or validation.
        if spec.kind == "attention":
            if self.direct:
                descriptor = spec.direct_spec
                self.output_projection = None
                self.output_base = descriptor.output_base
                self.output_base_weight = descriptor.output_base.weight
                self.output_base_bias = descriptor.output_base.bias
                self.output_a_weight = descriptor.output_lora_a
                self.output_b_weight = descriptor.output_lora_b
                self.output_scaling = descriptor.output_scaling
                self.output_adapter_dtype = descriptor.output_lora_a.dtype
                self.output_cast_adapter_input = True
            else:
                output = spec.output_projection
                if output is None:  # pragma: no cover
                    raise RuntimeError(
                        f"{spec.label}: output projection is missing"
                    )
                self.output_projection = output
                self.output_base = output.base
                self.output_base_weight = output.base.weight
                self.output_base_bias = output.base.bias
                self.output_a_weight = output.lora_a.weight
                self.output_b_weight = output.lora_b.weight
                self.output_scaling = output.scaling
                self.output_adapter_dtype = output.lora_a.weight.dtype
                self.output_cast_adapter_input = (
                    output.cast_input_dtype_enabled
                )
            self.head_dim = self.parent.head_dim
            self.layer_index = self.parent.layer_idx
            self.attention_scaling = self.parent.scaling
            self.sliding_window = self.parent.sliding_window
            self.rotary = spec.pinned.apply_rotary_pos_emb
            self.attention_interface = spec.attention_interface
            self.down_weight = None
        elif spec.kind == "mlp":
            if spec.down_projection is None:  # pragma: no cover
                raise RuntimeError(f"{spec.label}: down projection is missing")
            self.output_projection = None
            self.output_base = None
            self.output_base_weight = None
            self.output_base_bias = None
            self.output_a_weight = None
            self.output_b_weight = None
            self.output_scaling = None
            self.output_adapter_dtype = None
            self.output_cast_adapter_input = None
            self.head_dim = None
            self.layer_index = None
            self.attention_scaling = None
            self.sliding_window = None
            self.rotary = None
            self.attention_interface = None
            self.down_weight = spec.down_projection.weight
        else:  # pragma: no cover - authenticated by internal construction.
            raise RuntimeError(f"unsupported parent kind: {spec.kind}")

        for attribute in (
            _PACKED_BASE_WEIGHT,
            _PACKED_BASE_BIAS,
            _PACKED_LORA_A_WEIGHT,
        ):
            if hasattr(self.parent, attribute):
                raise RuntimeError(
                    f"{self.spec.label}: packed parent attribute already exists"
                )

        packed_base_weight = torch.cat(
            [weight.detach() for weight in self.base_weights],
            dim=0,
        )
        if self.base_biases[0] is None:
            packed_base_bias = None
        else:
            packed_base_bias = torch.cat(
                [bias.detach() for bias in self.base_biases],
                dim=0,
            )
        packed_lora_a = (
            spec.direct_spec.packed_lora_a
            if self.direct
            else torch.nn.Parameter(
                torch.cat(
                    [weight.detach() for weight in self.a_weights],
                    dim=0,
                ),
                requires_grad=True,
            )
        )
        self.packed_base_weight = packed_base_weight
        self.packed_base_bias = packed_base_bias
        self.packed_lora_a = packed_lora_a
        self.base_weight_views = tuple(
            packed_base_weight.detach().split(self.output_widths, dim=0)
        )
        self.base_bias_views = (
            ()
            if packed_base_bias is None
            else tuple(
                packed_base_bias.detach().split(self.output_widths, dim=0)
            )
        )
        self.a_weight_views = (
            ()
            if self.direct
            else tuple(packed_lora_a.detach().split(self.ranks, dim=0))
        )

        try:
            self.parent.register_buffer(
                _PACKED_BASE_WEIGHT,
                packed_base_weight,
                persistent=False,
            )
            self.parent.register_buffer(
                _PACKED_BASE_BIAS,
                packed_base_bias,
                persistent=False,
            )
            if not self.direct:
                self.parent.register_parameter(
                    _PACKED_LORA_A_WEIGHT,
                    packed_lora_a,
                )

            # Repoint, rather than retain, the original Parameter storage.  The
            # objects survive for exact identity restoration, while their old
            # allocations are released as soon as set_ succeeds.
            for parameter, view in zip(
                self.base_weights,
                self.base_weight_views,
            ):
                self._alias_parameter_storage(parameter, view, "base weight")
            for parameter, view in zip(self.a_weights, self.a_weight_views):
                self._alias_parameter_storage(parameter, view, "LoRA-A weight")
            for parameter, view in zip(
                self.base_biases,
                self.base_bias_views,
            ):
                self._alias_parameter_storage(parameter, view, "base bias")

            for base in self.base_layers:
                delattr(base, "weight")
                delattr(base, "bias")
            for lora_a in self.a_layers:
                delattr(lora_a, "weight")
            self._state_hook = self.parent.register_state_dict_pre_hook(
                self._reject_packed_state_dict
            )
            forward = (
                self._attention_forward
                if self.spec.kind == "attention"
                else self._mlp_forward
            )
            installed = MethodType(forward, self.parent)
            self.parent.forward = installed
            self._installed_forward = installed
        except Exception:
            self._rollback_install()
            raise

    @staticmethod
    def _same_storage_view(parameter, view) -> bool:
        return (
            parameter.dtype == view.dtype
            and parameter.device == view.device
            and parameter.shape == view.shape
            and parameter.stride() == view.stride()
            and parameter.storage_offset() == view.storage_offset()
            and parameter.untyped_storage().data_ptr()
            == view.untyped_storage().data_ptr()
        )

    def _alias_parameter_storage(self, parameter, view, description) -> None:
        if parameter is None:
            raise RuntimeError(
                f"{self.spec.label}: cannot alias an absent {description}"
            )
        requires_grad = parameter.requires_grad
        with torch.no_grad():
            parameter.set_(view.detach())
        if parameter.requires_grad != requires_grad:
            raise RuntimeError(
                f"{self.spec.label}: {description} trainability changed"
            )
        if not self._same_storage_view(parameter, view):
            raise RuntimeError(
                f"{self.spec.label}: {description} storage alias failed"
            )

    def _rollback_install(self) -> None:
        if (
            self._installed_forward is not None
            and self.parent.__dict__.get("forward") is self._installed_forward
        ):
            delattr(self.parent, "forward")
        if self._state_hook is not None:
            self._state_hook.remove()
            self._state_hook = None
        # A failed install must not leave the ordinary modules sharing packed
        # storage without the serialization guard.
        with torch.no_grad():
            for parameter in (
                *self.base_weights,
                *self.a_weights,
                *(bias for bias in self.base_biases if bias is not None),
            ):
                parameter.set_(
                    parameter.detach().clone(
                        memory_format=torch.contiguous_format
                    )
                )
        for attribute, registry in (
            (_PACKED_LORA_A_WEIGHT, self.parent._parameters),
            (_PACKED_BASE_BIAS, self.parent._buffers),
            (_PACKED_BASE_WEIGHT, self.parent._buffers),
        ):
            if attribute in registry:
                delattr(self.parent, attribute)
        for base, weight, bias in zip(
            self.base_layers,
            self.base_weights,
            self.base_biases,
        ):
            if "weight" not in base._parameters:
                base.register_parameter("weight", weight)
            if "bias" not in base._parameters:
                base.register_parameter("bias", bias)
        for lora_a, weight in zip(self.a_layers, self.a_weights):
            if "weight" not in lora_a._parameters:
                lora_a.register_parameter("weight", weight)

    def _reject_packed_state_dict(self, _module, _prefix, _keep_vars) -> None:
        raise RuntimeError(
            f"{self.spec.label}: refusing to serialize temporary parent-packed "
            "projections; call materialize_standard_peft() first"
        )

    def _validate_projection_pre_materialization(
        self,
        projection: _Projection,
        *,
        packed: bool,
    ) -> None:
        wrapper = projection.wrapper
        if tuple(wrapper.active_adapters) != (ADAPTER_NAME,):
            raise RuntimeError(
                f"{projection.name}: active adapter changed after installation"
            )
        if wrapper.disable_adapters is not False or wrapper.merged is not False:
            raise RuntimeError(
                f"{projection.name}: adapter state changed after installation"
            )
        if wrapper.lora_variant != {}:
            raise RuntimeError(
                f"{projection.name}: LoRA variant appeared after installation"
            )
        if (
            _mapping_member(wrapper.lora_A, f"{projection.name}.lora_A")
            is not projection.lora_a
            or _mapping_member(wrapper.lora_B, f"{projection.name}.lora_B")
            is not projection.lora_b
            or _mapping_member(
                wrapper.lora_dropout,
                f"{projection.name}.lora_dropout",
            )
            is not projection.dropout
            or _mapping_member(wrapper.scaling, f"{projection.name}.scaling")
            != projection.scaling
        ):
            raise RuntimeError(f"{projection.name}: LoRA branch identity changed")
        if (
            wrapper.cast_input_dtype_enabled
            != projection.cast_input_dtype_enabled
        ):
            raise RuntimeError(f"{projection.name}: input-cast state changed")
        if wrapper.base_layer is not projection.base:
            raise RuntimeError(f"{projection.name}: base-layer identity changed")
        if packed:
            if (
                "weight" in projection.base._parameters
                or "bias" in projection.base._parameters
                or "weight" in projection.lora_a._parameters
            ):
                raise RuntimeError(
                    f"{projection.name}: packed representation changed"
                )
        elif (
            projection.base.weight is not self.output_base_weight
            or projection.base.bias is not self.output_base_bias
            or projection.lora_a.weight is not self.output_a_weight
            or projection.lora_b.weight is not self.output_b_weight
        ):
            raise RuntimeError(
                f"{projection.name}: cached output tensors changed"
            )
        for bypassed, suffix in (
            (wrapper, ""),
            (projection.base, ".base_layer"),
            (projection.lora_a, ".lora_A.default"),
            (projection.lora_b, ".lora_B.default"),
            (projection.dropout, ".lora_dropout.default"),
        ):
            _require_no_local_call_hooks(
                bypassed,
                f"{projection.name}{suffix}",
            )

    def _validate_pre_materialization(self) -> None:
        if self._materialized:
            raise RuntimeError(
                f"{self.spec.label}: stale packed group after materialization"
            )
        if self.parent.__dict__.get("forward") is not self._installed_forward:
            raise RuntimeError(f"{self.spec.label}: parent forward changed")
        if (
            _PACKED_BASE_WEIGHT not in self.parent._buffers
            or _PACKED_BASE_BIAS not in self.parent._buffers
            or getattr(self.parent, _PACKED_BASE_WEIGHT, None)
            is not self.packed_base_weight
            or getattr(self.parent, _PACKED_BASE_BIAS, None)
            is not self.packed_base_bias
        ):
            raise RuntimeError(f"{self.spec.label}: packed tensors changed")
        if (
            self._state_hook is None
            or self._state_hook.id not in self.parent._state_dict_pre_hooks
        ):
            raise RuntimeError(f"{self.spec.label}: state-dict guard changed")
        if self.direct:
            try:
                current_descriptor = (
                    self.spec.direct_adapter.parent_dispatch_spec(
                        self.spec.layer_index,
                        self.spec.kind,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{self.spec.label}: direct descriptor became unavailable"
                ) from exc
            if current_descriptor is not self.spec.direct_spec:
                raise RuntimeError(
                    f"{self.spec.label}: direct descriptor identity changed"
                )
            _require_direct_registration(
                self.parent,
                (self.packed_lora_a, *self.b_weights),
                self.spec.label,
            )
        elif (
            _PACKED_LORA_A_WEIGHT not in self.parent._parameters
            or getattr(self.parent, _PACKED_LORA_A_WEIGHT, None)
            is not self.packed_lora_a
        ):
            raise RuntimeError(f"{self.spec.label}: packed LoRA-A changed")
        if (
            self.packed_base_weight.shape
            != (sum(self.output_widths), self.input_features)
            or self.packed_base_weight.dtype != self.base_dtype
            or self.packed_base_weight.device != self.device
            or self.packed_base_weight.requires_grad
        ):
            raise RuntimeError(f"{self.spec.label}: packed base weight changed")
        if self.base_biases[0] is None:
            if self.packed_base_bias is not None:
                raise RuntimeError(f"{self.spec.label}: packed base bias changed")
        elif (
            self.packed_base_bias is None
            or self.packed_base_bias.shape != (sum(self.output_widths),)
            or self.packed_base_bias.dtype != self.base_dtype
            or self.packed_base_bias.device != self.device
            or self.packed_base_bias.requires_grad
        ):
            raise RuntimeError(f"{self.spec.label}: packed base bias changed")
        if (
            self.packed_lora_a.shape
            != (sum(self.ranks), self.input_features)
            or self.packed_lora_a.dtype != self.adapter_dtype
            or self.packed_lora_a.device != self.device
            or not self.packed_lora_a.is_contiguous()
        ):
            raise RuntimeError(f"{self.spec.label}: packed LoRA-A changed")

        member_names = (
            ("q_proj", "k_proj", "v_proj")
            if self.spec.kind == "attention"
            else ("gate_proj", "up_proj")
        )
        for index, (member_name, base, weight) in enumerate(
            zip(member_names, self.base_layers, self.base_weights)
        ):
            expected_member = (
                base if self.direct else self.projections[index].wrapper
            )
            if getattr(self.parent, member_name) is not expected_member:
                raise RuntimeError(
                    f"{self.spec.label}: {member_name} identity changed"
                )
            if not self._same_storage_view(
                weight,
                self.base_weight_views[index],
            ):
                raise RuntimeError(
                    f"{self.spec.label}: cached {member_name} base changed"
                )
            if self.base_biases[index] is not None and not self._same_storage_view(
                self.base_biases[index],
                self.base_bias_views[index],
            ):
                raise RuntimeError(
                    f"{self.spec.label}: cached {member_name} bias changed"
                )
        if self.direct:
            descriptor = self.spec.direct_spec
            if (
                len(tuple(descriptor.base_layers)) != len(self.base_layers)
                or any(
                    observed is not expected
                    for observed, expected in zip(
                        tuple(descriptor.base_layers),
                        self.base_layers,
                    )
                )
                or descriptor.packed_lora_a is not self.packed_lora_a
                or len(tuple(descriptor.lora_b_weights))
                != len(self.b_weights)
                or any(
                    observed is not expected
                    for observed, expected in zip(
                        tuple(descriptor.lora_b_weights),
                        self.b_weights,
                    )
                )
                or tuple(descriptor.scalings) != self.scalings
            ):
                raise RuntimeError(
                    f"{self.spec.label}: direct cached inventory changed"
                )
            for index, (weight, width, rank) in enumerate(
                zip(self.b_weights, self.output_widths, self.ranks)
            ):
                if (
                    weight.shape != (width, rank)
                    or weight.dtype != self.adapter_dtype
                    or weight.device != self.device
                    or not weight.is_contiguous()
                ):
                    raise RuntimeError(
                        f"{self.spec.label}: direct LoRA-B {index} changed"
                    )
        else:
            for index, (projection, a_weight, b_weight) in enumerate(
                zip(self.projections, self.a_weights, self.b_weights)
            ):
                if (
                    not self._same_storage_view(
                        a_weight,
                        self.a_weight_views[index],
                    )
                    or projection.lora_b.weight is not b_weight
                    or b_weight.shape
                    != (projection.out_features, projection.rank)
                    or b_weight.dtype != self.adapter_dtype
                    or b_weight.device != self.device
                ):
                    raise RuntimeError(
                        f"{projection.name}: cached adapter tensor changed"
                    )
                self._validate_projection_pre_materialization(
                    projection,
                    packed=True,
                )

        if self.spec.kind == "attention":
            if (
                self.parent.head_dim != self.head_dim
                or self.parent.layer_idx != self.layer_index
                or self.parent.scaling != self.attention_scaling
                or self.parent.sliding_window != self.sliding_window
                or self.parent.attention_dropout != 0.0
            ):
                raise RuntimeError(
                    f"{self.spec.label}: cached attention scalars changed"
                )
            if self.direct:
                descriptor = self.spec.direct_spec
                if (
                    self.parent.o_proj is not self.output_base
                    or descriptor.output_base is not self.output_base
                    or descriptor.output_lora_a is not self.output_a_weight
                    or descriptor.output_lora_b is not self.output_b_weight
                    or descriptor.output_scaling != self.output_scaling
                    or self.output_base.weight is not self.output_base_weight
                    or self.output_base.bias is not self.output_base_bias
                    or self.output_base_weight.shape
                    != (
                        self.output_base.out_features,
                        self.output_base.in_features,
                    )
                    or self.output_base_weight.dtype != self.base_dtype
                    or self.output_base_weight.device != self.device
                    or not self.output_base_weight.is_contiguous()
                    or self.output_base_weight.requires_grad
                ):
                    raise RuntimeError(
                        f"{self.spec.label}: direct output inventory changed"
                    )
                output_rank = self.output_a_weight.shape[0]
                if (
                    self.output_a_weight.shape
                    != (output_rank, self.output_base.in_features)
                    or self.output_b_weight.shape
                    != (self.output_base.out_features, output_rank)
                    or self.output_a_weight.dtype
                    != self.output_adapter_dtype
                    or self.output_b_weight.dtype
                    != self.output_adapter_dtype
                    or self.output_a_weight.device != self.device
                    or self.output_b_weight.device != self.device
                    or not self.output_a_weight.is_contiguous()
                    or not self.output_b_weight.is_contiguous()
                ):
                    raise RuntimeError(
                        f"{self.spec.label}: direct output tensors changed"
                    )
                _require_direct_registration(
                    self.parent,
                    (self.output_a_weight, self.output_b_weight),
                    self.spec.label,
                )
                _require_no_local_call_hooks(
                    self.output_base,
                    f"{self.spec.label} o_proj",
                )
            else:
                output = self.output_projection
                if self.parent.o_proj is not output.wrapper:
                    raise RuntimeError(
                        f"{self.spec.label}: output projection identity changed"
                    )
                self._validate_projection_pre_materialization(
                    output,
                    packed=False,
                )
            resolved_interface = (
                self.spec.pinned.attention_interfaces.get_interface(
                    self.parent.config._attn_implementation,
                    self.spec.pinned.eager_attention_forward,
                )
            )
            if resolved_interface is not self.attention_interface:
                raise RuntimeError(
                    f"{self.spec.label}: attention interface changed"
                )
        else:
            down = self.spec.down_projection
            if (
                self.parent.down_proj is not down
                or down.weight is not self.down_weight
                or down.bias is not None
                or down.weight.shape
                != (down.out_features, down.in_features)
                or down.weight.dtype != self.base_dtype
                or down.weight.device != self.device
                or not down.weight.is_contiguous()
                or down.weight.requires_grad
            ):
                raise RuntimeError(
                    f"{self.spec.label}: cached down projection changed"
                )
            _require_no_local_call_hooks(down, f"{self.spec.label} down_proj")
            _require_no_local_call_hooks(
                self.parent.act_fn,
                f"{self.spec.label} activation",
            )

    def _project(self, hidden_states):
        projected_base = torch.nn.functional.linear(
            hidden_states,
            self.packed_base_weight,
            self.packed_base_bias,
        )
        adapter_input = (
            hidden_states.to(dtype=self.adapter_dtype)
            if self.cast_adapter_input
            else hidden_states
        )
        projected_a = torch.nn.functional.linear(
            adapter_input,
            self.packed_lora_a,
        )
        base_pieces = projected_base.split(self.output_widths, dim=-1)
        a_pieces = projected_a.split(self.ranks, dim=-1)
        return tuple(
            (
                base_piece + _LINEAR(a_piece, b_weight) * scaling
            ).to(base_piece.dtype)
            for base_piece, a_piece, b_weight, scaling in zip(
                base_pieces,
                a_pieces,
                self.b_weights,
                self.scalings,
            )
        )

    def _project_output(self, hidden_states):
        projected_base = _LINEAR(
            hidden_states,
            self.output_base_weight,
            self.output_base_bias,
        )
        adapter_input = (
            hidden_states.to(dtype=self.output_adapter_dtype)
            if self.output_cast_adapter_input
            else hidden_states
        )
        projected_a = _LINEAR(adapter_input, self.output_a_weight)
        projected_b = _LINEAR(projected_a, self.output_b_weight)
        return (
            projected_base + projected_b * self.output_scaling
        ).to(projected_base.dtype)

    def _attention_forward(
        group,
        attention,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, group.head_dim)

        query_states, key_states, value_states = group._project(hidden_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = group.rotary(
            query_states,
            key_states,
            cos,
            sin,
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                group.layer_index,
            )

        attn_output, attn_weights = group.attention_interface(
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=group.attention_scaling,
            sliding_window=group.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return group._project_output(attn_output), attn_weights

    def _mlp_forward(group, _mlp, x):
        gate, up = group._project(x)
        activated = _SILU(gate) * up
        return _LINEAR(activated, group.down_weight)

    def preflight_materialization(self) -> None:
        if self._materialized:
            return
        self._validate_pre_materialization()

    def materialize_standard_peft(self) -> None:
        if self._materialized:
            return
        self.preflight_materialization()
        packed_requires_grad = self.packed_lora_a.requires_grad

        # Restore independent standard Linear storage one group at a time.  This
        # keeps the original Parameter identities while avoiding shared-storage
        # surprises in safetensors after the temporary packed object is removed.
        with torch.no_grad():
            for original, view in zip(
                self.base_weights,
                self.base_weight_views,
            ):
                original.set_(view.clone(memory_format=torch.contiguous_format))
            for original, view in zip(self.a_weights, self.a_weight_views):
                original.set_(view.clone(memory_format=torch.contiguous_format))
            for original, view in zip(
                self.base_biases,
                self.base_bias_views,
            ):
                original.set_(view.clone(memory_format=torch.contiguous_format))

        delattr(self.parent, "forward")
        if not self.direct:
            delattr(self.parent, _PACKED_LORA_A_WEIGHT)
        delattr(self.parent, _PACKED_BASE_BIAS)
        delattr(self.parent, _PACKED_BASE_WEIGHT)
        for base, weight, bias in zip(
            self.base_layers,
            self.base_weights,
            self.base_biases,
        ):
            base.register_parameter("weight", weight)
            base.register_parameter("bias", bias)
        for lora_a, weight in zip(self.a_layers, self.a_weights):
            weight.requires_grad_(packed_requires_grad)
            lora_a.register_parameter("weight", weight)
        self._state_hook.remove()
        self._state_hook = None
        self._installed_forward = None
        self.packed_base_weight = None
        self.packed_base_bias = None
        self.packed_lora_a = None
        self.base_weight_views = ()
        self.base_bias_views = ()
        self.a_weight_views = ()
        self._materialized = True


class ParentLayerDispatchPlan:
    """Own every installed parent dispatch until standard export materialization."""

    def __init__(self, specs: tuple[_GroupSpec, ...]) -> None:
        installed = []
        try:
            for spec in specs:
                installed.append(_PackedParent(spec))
        except Exception:
            for group in reversed(installed):
                group.materialize_standard_peft()
            raise
        self.groups = tuple(installed)
        self.attention_groups = tuple(
            group for group in self.groups if group.spec.kind == "attention"
        )
        self.mlp_groups = tuple(
            group for group in self.groups if group.spec.kind == "mlp"
        )
        self._materialized = False

    @property
    def materialized(self) -> bool:
        return self._materialized

    def materialize_standard_peft(self) -> None:
        if self._materialized:
            return
        for group in self.groups:
            group.preflight_materialization()
        for group in reversed(self.groups):
            group.materialize_standard_peft()
        self._materialized = True

    def close(self) -> None:
        self.materialize_standard_peft()


def install_parent_layer_dispatch(
    transformer,
    direct_adapter=None,
) -> ParentLayerDispatchPlan:
    """Install the exact 28-layer Qwen2 parent projection dispatch.

    Complete structural validation happens before the first parameter is moved.
    ``direct_adapter`` may be the wrapper-free descriptor returned by this
    candidate's direct packed injector. The returned plan must be materialized
    before serialization.
    """

    specs = _validate_transformer(
        transformer,
        direct_adapter=direct_adapter,
    )
    return ParentLayerDispatchPlan(specs)
