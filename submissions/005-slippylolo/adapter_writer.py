"""Direct, standard PEFT LoRA artifact writer for static injection variants."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
EXPECTED_FILES = frozenset({ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME})
EXPECTED_QWEN2_LAYERS = 28
PROJECTIONS = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
)
EXPECTED_TARGET_MODULES = tuple(
    projection_name for _owner_name, projection_name in PROJECTIONS
)


def _empty_output_directory(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    entries = tuple(destination.iterdir())
    if entries:
        raise RuntimeError(
            "adapter output directory must be empty before direct serialization"
        )
    return destination


def _adapter_state(transformer):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - benchmark runtime dependency
        raise RuntimeError("torch is required to serialize the direct adapter") from exc

    try:
        layers = tuple(transformer.layers)
    except AttributeError as exc:
        raise RuntimeError("pinned Qwen2 decoder structure is unavailable") from exc
    if len(layers) != EXPECTED_QWEN2_LAYERS:
        raise RuntimeError(
            f"direct writer requires exactly {EXPECTED_QWEN2_LAYERS} decoder layers"
        )

    tensors = {}
    parameter_ids = set()
    for layer_index, layer in enumerate(layers):
        for owner_name, projection_name in PROJECTIONS:
            try:
                projection = getattr(getattr(layer, owner_name), projection_name)
                adapter = tuple(projection.active_adapters)
                lora_a = projection.lora_A["default"].weight
                lora_b = projection.lora_B["default"].weight
            except (AttributeError, KeyError) as exc:
                raise RuntimeError(
                    f"layer {layer_index} {owner_name}.{projection_name} "
                    "is not a standard single-adapter LoRA projection"
                ) from exc
            if adapter != ("default",):
                raise RuntimeError("direct writer requires only the default adapter")
            if id(lora_a) in parameter_ids or id(lora_b) in parameter_ids:
                raise RuntimeError("direct writer found a shared LoRA parameter")
            parameter_ids.update((id(lora_a), id(lora_b)))
            if (
                lora_a.ndim != 2
                or lora_b.ndim != 2
                or lora_a.shape[0] != lora_b.shape[1]
                or lora_a.dtype != torch.bfloat16
                or lora_b.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "direct writer requires compatible two-dimensional BF16 LoRA"
                )
            prefix = (
                f"base_model.model.model.layers.{layer_index}."
                f"{owner_name}.{projection_name}"
            )
            tensors[f"{prefix}.lora_A.weight"] = (
                lora_a.detach().to(device="cpu").contiguous()
            )
            tensors[f"{prefix}.lora_B.weight"] = (
                lora_b.detach().to(device="cpu").contiguous()
            )

    if len(tensors) != EXPECTED_QWEN2_LAYERS * len(PROJECTIONS) * 2:
        raise RuntimeError("direct writer tensor inventory changed unexpectedly")
    return tensors


def _direct_packed_adapter_state(transformer, packed_injection):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - benchmark runtime dependency
        raise RuntimeError("torch is required to serialize the direct adapter") from exc

    if getattr(packed_injection, "transformer", None) is not transformer:
        raise RuntimeError("direct packed adapter belongs to a different transformer")
    adapter_layers = tuple(getattr(packed_injection, "layers", ()))
    try:
        transformer_layers = tuple(transformer.layers)
    except AttributeError as exc:
        raise RuntimeError("pinned Qwen2 decoder structure is unavailable") from exc
    if (
        len(adapter_layers) != EXPECTED_QWEN2_LAYERS
        or len(transformer_layers) != EXPECTED_QWEN2_LAYERS
    ):
        raise RuntimeError(
            "direct packed writer requires exactly "
            f"{EXPECTED_QWEN2_LAYERS} adapter layers"
        )
    rank = getattr(packed_injection, "rank", None)
    alpha = getattr(packed_injection, "alpha", None)
    if type(rank) is not int or rank < 1:
        raise RuntimeError("direct packed writer requires a positive integer rank")
    if type(alpha) is not int or alpha < 1:
        raise RuntimeError("direct packed writer requires a positive integer alpha")
    expected_scaling = alpha / rank

    tensors = {}
    seen_parameters = set()
    for layer_index, (adapter_layer, transformer_layer) in enumerate(
        zip(adapter_layers, transformer_layers)
    ):
        attention = adapter_layer.attention
        mlp = adapter_layer.mlp
        if attention.kind != "attention" or mlp.kind != "mlp":
            raise RuntimeError("direct packed parent kinds changed")
        if (
            len(attention.base_layers) != 3
            or len(attention.lora_b_weights) != 3
            or len(mlp.base_layers) != 2
            or len(mlp.lora_b_weights) != 2
            or attention.output_base is None
            or attention.output_lora_a is None
            or attention.output_lora_b is None
        ):
            raise RuntimeError("direct packed adapter inventory is incomplete")

        actual_qkv = (
            transformer_layer.self_attn.q_proj,
            transformer_layer.self_attn.k_proj,
            transformer_layer.self_attn.v_proj,
        )
        actual_output = transformer_layer.self_attn.o_proj
        actual_gate_up = (
            transformer_layer.mlp.gate_proj,
            transformer_layer.mlp.up_proj,
        )
        if (
            any(
                specified is not actual
                for specified, actual in zip(attention.base_layers, actual_qkv)
            )
            or attention.output_base is not actual_output
            or any(
                specified is not actual
                for specified, actual in zip(mlp.base_layers, actual_gate_up)
            )
        ):
            raise RuntimeError(
                "direct packed adapter base projection order changed"
            )
        if any(
            type(base) is not torch.nn.Linear
            for base in (*actual_qkv, actual_output, *actual_gate_up)
        ):
            raise RuntimeError(
                "direct packed writer requires exact nn.Linear base projections"
            )
        if (
            tuple(attention.scalings) != (expected_scaling,) * 3
            or attention.output_scaling != expected_scaling
            or tuple(mlp.scalings) != (expected_scaling,) * 2
        ):
            raise RuntimeError(
                "direct packed adapter scaling differs from rank/alpha"
            )

        adapter_parameters = (
            attention.packed_lora_a,
            *attention.lora_b_weights,
            attention.output_lora_a,
            attention.output_lora_b,
            mlp.packed_lora_a,
            *mlp.lora_b_weights,
        )
        if any(
            not isinstance(parameter, torch.nn.Parameter)
            for parameter in adapter_parameters
        ):
            raise RuntimeError(
                "direct packed adapter inventory contains a non-Parameter"
            )
        parameter_ids = tuple(id(parameter) for parameter in adapter_parameters)
        if (
            len(set(parameter_ids)) != len(parameter_ids)
            or any(identity in seen_parameters for identity in parameter_ids)
        ):
            raise RuntimeError("direct packed writer found a shared LoRA parameter")
        seen_parameters.update(parameter_ids)

        if (
            attention.packed_lora_a.shape
            != (3 * rank, actual_qkv[0].in_features)
            or any(weight.shape[1] != rank for weight in attention.lora_b_weights)
            or attention.output_lora_a.shape
            != (rank, actual_output.in_features)
            or attention.output_lora_b.shape
            != (actual_output.out_features, rank)
            or mlp.packed_lora_a.shape
            != (2 * rank, actual_gate_up[0].in_features)
            or any(weight.shape[1] != rank for weight in mlp.lora_b_weights)
        ):
            raise RuntimeError(
                "direct packed adapter rank differs from its export config"
            )

        qkv_a = attention.packed_lora_a.detach().split(rank, dim=0)
        gate_up_a = mlp.packed_lora_a.detach().split(rank, dim=0)
        projections = (
            ("self_attn", "q_proj", qkv_a[0], attention.lora_b_weights[0]),
            ("self_attn", "k_proj", qkv_a[1], attention.lora_b_weights[1]),
            ("self_attn", "v_proj", qkv_a[2], attention.lora_b_weights[2]),
            (
                "self_attn",
                "o_proj",
                attention.output_lora_a,
                attention.output_lora_b,
            ),
            ("mlp", "gate_proj", gate_up_a[0], mlp.lora_b_weights[0]),
            ("mlp", "up_proj", gate_up_a[1], mlp.lora_b_weights[1]),
        )
        for owner_name, projection_name, lora_a, lora_b in projections:
            base = getattr(
                getattr(transformer_layer, owner_name),
                projection_name,
            )
            if (
                lora_a.ndim != 2
                or lora_b.ndim != 2
                or lora_a.shape[0] != lora_b.shape[1]
                or lora_a.shape[1] != base.in_features
                or lora_b.shape[0] != base.out_features
                or lora_a.dtype != torch.bfloat16
                or lora_b.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "direct packed writer requires compatible BF16 LoRA slices"
                )
            prefix = (
                f"base_model.model.model.layers.{layer_index}."
                f"{owner_name}.{projection_name}"
            )
            tensors[f"{prefix}.lora_A.weight"] = (
                lora_a.detach().to(device="cpu").contiguous()
            )
            tensors[f"{prefix}.lora_B.weight"] = (
                lora_b.detach().to(device="cpu").contiguous()
            )

    if len(tensors) != EXPECTED_QWEN2_LAYERS * len(PROJECTIONS) * 2:
        raise RuntimeError(
            "direct packed writer tensor inventory changed unexpectedly"
        )
    return tensors


def write_static_lora_adapter(
    *,
    transformer,
    output_dir: str | Path,
    base_model_name_or_path: str,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    packed_injection=None,
) -> None:
    """Write exactly the two files consumed by ``PeftModel.from_pretrained``."""

    from safetensors.torch import save_file

    resolved_targets = tuple(target_modules)
    if resolved_targets != EXPECTED_TARGET_MODULES:
        raise RuntimeError(
            "direct writer target order must exactly match the pinned projections"
        )
    if packed_injection is not None and (
        getattr(packed_injection, "rank", None) != rank
        or getattr(packed_injection, "alpha", None) != alpha
    ):
        raise RuntimeError(
            "direct packed adapter rank/alpha differ from export config"
        )
    destination = _empty_output_directory(output_dir)
    if packed_injection is None:
        tensors = _adapter_state(transformer)
    else:
        tensors = _direct_packed_adapter_state(
            transformer,
            packed_injection,
        )
    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(resolved_targets),
        "task_type": "CAUSAL_LM",
    }
    (destination / ADAPTER_CONFIG_NAME).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(
        tensors,
        str(destination / ADAPTER_WEIGHTS_NAME),
        metadata={"format": "pt"},
    )
    assert_standard_two_file_artifact(destination)


def normalize_peft_artifact(output_dir: str | Path) -> None:
    """Remove only PEFT's generated model card and authenticate the artifact."""

    destination = Path(output_dir)
    readme = destination / "README.md"
    if readme.exists():
        readme.unlink()
    assert_standard_two_file_artifact(destination)


def assert_standard_two_file_artifact(output_dir: str | Path) -> None:
    destination = Path(output_dir)
    observed = frozenset(
        entry.name for entry in destination.iterdir() if entry.is_file()
    )
    if observed != EXPECTED_FILES or any(
        not entry.is_file() for entry in destination.iterdir()
    ):
        raise RuntimeError(
            "adapter artifact must contain exactly adapter_config.json and "
            f"adapter_model.safetensors; got {sorted(observed)!r}"
        )
