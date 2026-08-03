"""Authenticated low-level loader for the pinned Qwen2.5-1.5B snapshot.

This module deliberately avoids the Hugging Face Auto* and from_pretrained paths.
It keeps the exact pinned Qwen implementation class so the existing direct-packed
LoRA runtime and its structural guards remain applicable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import torch


EXPECTED_CONFIG = {
    "model_type": "qwen2",
    "vocab_size": 151_936,
    "hidden_size": 1_536,
    "intermediate_size": 8_960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "max_position_embeddings": 131_072,
    "max_window_layers": 28,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "sliding_window": 131_072,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_mrope": False,
    "use_sliding_window": False,
    "tie_word_embeddings": True,
    "bos_token_id": 151_643,
    "eos_token_id": 151_643,
}
EXPECTED_EOS_TOKEN = "<|endoftext|>"
EXPECTED_TOKENIZER_SIZE = 151_665
EXPECTED_WEIGHTS_NAME = "model.safetensors"
EXPECTED_TOKENIZER_NAME = "tokenizer.json"
EXPECTED_CONFIG_NAME = "config.json"
EXPECTED_GENERATION_CONFIG_NAME = "generation_config.json"


class PinnedLoaderError(RuntimeError):
    """The local snapshot or its direct runtime differs from the pinned contract."""


def _phase(
    emit: Callable[..., None],
    name: str,
    started: float,
    **fields: object,
) -> None:
    emit(
        name,
        duration_seconds=time.monotonic() - started,
        **fields,
    )


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PinnedLoaderError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PinnedLoaderError(f"{label} must be a JSON object")
    return value


def authenticate_snapshot(snapshot: str | Path) -> tuple[Path, dict[str, object]]:
    root = Path(snapshot)
    if not root.is_dir():
        raise PinnedLoaderError("pinned Qwen snapshot directory is unavailable")
    required = {
        EXPECTED_CONFIG_NAME,
        EXPECTED_GENERATION_CONFIG_NAME,
        EXPECTED_TOKENIZER_NAME,
        EXPECTED_WEIGHTS_NAME,
    }
    missing = tuple(name for name in sorted(required) if not (root / name).is_file())
    if missing:
        raise PinnedLoaderError(
            "pinned Qwen snapshot is missing required files: " + ", ".join(missing)
        )
    if tuple(root.glob("model-*.safetensors")) or tuple(
        root.glob("*.safetensors.index.json")
    ):
        raise PinnedLoaderError(
            "pinned Qwen snapshot unexpectedly uses a sharded weight layout"
        )

    raw_config = _read_json_object(root / EXPECTED_CONFIG_NAME, "model config")
    mismatches = tuple(
        key
        for key, expected in EXPECTED_CONFIG.items()
        if raw_config.get(key) != expected
    )
    if mismatches:
        raise PinnedLoaderError(
            "pinned Qwen model configuration differs: " + ", ".join(mismatches)
        )
    if raw_config.get("torch_dtype") != "bfloat16":
        raise PinnedLoaderError("pinned Qwen model dtype is not BF16")

    generation = _read_json_object(
        root / EXPECTED_GENERATION_CONFIG_NAME,
        "generation config",
    )
    if generation.get("eos_token_id") != EXPECTED_CONFIG["eos_token_id"]:
        raise PinnedLoaderError(
            "generation config EOS differs from the pinned model config"
        )
    return root, raw_config


def load_direct_tokenizer(
    snapshot: str | Path,
    *,
    emit: Callable[..., None],
):
    root, _raw_config = authenticate_snapshot(snapshot)
    import_started = time.monotonic()
    from tokenizers import Tokenizer

    _phase(emit, "tokenizers_import", import_started)
    load_started = time.monotonic()
    tokenizer = Tokenizer.from_file(str(root / EXPECTED_TOKENIZER_NAME))
    if tokenizer.get_vocab_size(with_added_tokens=True) != EXPECTED_TOKENIZER_SIZE:
        raise PinnedLoaderError("direct tokenizer vocabulary size differs")
    if tokenizer.token_to_id(EXPECTED_EOS_TOKEN) != EXPECTED_CONFIG["eos_token_id"]:
        raise PinnedLoaderError("direct tokenizer EOS identity differs")
    _phase(emit, "tokenizer_json_load", load_started)
    return tokenizer


def encode_text_batch(
    tokenizer,
    texts: Sequence[str],
    *,
    emit: Callable[..., None],
    phase_name: str,
) -> list[list[int]]:
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise PinnedLoaderError("direct tokenizer input must be a text sequence")
    if not all(isinstance(text, str) for text in texts):
        raise PinnedLoaderError("direct tokenizer input contains a non-string")
    started = time.monotonic()
    encodings = tokenizer.encode_batch(
        list(texts),
        add_special_tokens=False,
    )
    result = [list(encoding.ids) for encoding in encodings]
    if len(result) != len(texts):
        raise PinnedLoaderError("direct tokenizer output count differs")
    _phase(
        emit,
        phase_name,
        started,
        text_count=len(texts),
    )
    return result


def _recreate_rotary_buffers(model, config, rotary_type) -> None:
    rotary = rotary_type(config=config, device=torch.device("cuda", 0))
    if rotary.inv_freq.is_meta or rotary.original_inv_freq.is_meta:
        raise PinnedLoaderError("direct Qwen rotary buffers remained on meta")
    model.model.rotary_emb = rotary


def _assert_materialized_bf16_model(model) -> None:
    parameters = tuple(model.named_parameters())
    if not parameters:
        raise PinnedLoaderError("direct Qwen model has no parameters")
    meta_parameters = tuple(name for name, value in parameters if value.is_meta)
    if meta_parameters:
        raise PinnedLoaderError(
            "direct Qwen parameters remained on meta: "
            + ", ".join(meta_parameters[:4])
        )
    wrong_parameters = tuple(
        name
        for name, value in parameters
        if value.device.type != "cuda" or value.dtype != torch.bfloat16
    )
    if wrong_parameters:
        raise PinnedLoaderError(
            "direct Qwen parameter device/dtype differs: "
            + ", ".join(wrong_parameters[:4])
        )
    meta_buffers = tuple(
        name for name, value in model.named_buffers() if value.is_meta
    )
    if meta_buffers:
        raise PinnedLoaderError(
            "direct Qwen buffers remained on meta: " + ", ".join(meta_buffers[:4])
        )
    wrong_buffers = tuple(
        name
        for name, value in model.named_buffers()
        if value.device.type != "cuda"
    )
    if wrong_buffers:
        raise PinnedLoaderError(
            "direct Qwen buffer device differs: " + ", ".join(wrong_buffers[:4])
        )
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise PinnedLoaderError("direct Qwen tied embedding identity differs")


def load_direct_qwen(
    snapshot: str | Path,
    *,
    emit: Callable[..., None],
    warmer_done,
):
    root, raw_config = authenticate_snapshot(snapshot)
    safetensors_import_started = time.monotonic()
    from safetensors.torch import load_file

    _phase(emit, "safetensors_import", safetensors_import_started)
    transformers_import_started = time.monotonic()
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2ForCausalLM,
        Qwen2RotaryEmbedding,
    )

    _phase(emit, "transformers_qwen_import", transformers_import_started)

    config_started = time.monotonic()
    config = Qwen2Config.from_dict(dict(raw_config))
    config._attn_implementation = "sdpa"
    if config._attn_implementation != "sdpa":
        raise PinnedLoaderError("direct Qwen SDPA configuration was not retained")
    _phase(emit, "direct_qwen_config", config_started)

    construction_started = time.monotonic()
    cpu_rng_before = torch.get_rng_state()
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)
    cpu_rng_after = torch.get_rng_state()
    if not torch.equal(cpu_rng_before, cpu_rng_after):
        raise PinnedLoaderError("meta Qwen construction consumed CPU RNG state")
    _phase(emit, "direct_qwen_meta_construction", construction_started)

    load_started = time.monotonic()
    warmer_complete_at_start = bool(warmer_done.is_set())
    emit(
        "direct_safetensors_load_start",
        warmer_complete=warmer_complete_at_start,
    )
    state = load_file(
        str(root / EXPECTED_WEIGHTS_NAME),
        device="cuda:0",
    )
    incompatible = model.load_state_dict(
        state,
        strict=False,
        assign=True,
    )
    del state
    if tuple(incompatible.unexpected_keys):
        raise PinnedLoaderError(
            "direct Qwen weights contain unexpected keys: "
            + ", ".join(incompatible.unexpected_keys[:4])
        )
    if set(incompatible.missing_keys) not in (set(), {"lm_head.weight"}):
        raise PinnedLoaderError(
            "direct Qwen weights are missing unsupported keys: "
            + ", ".join(incompatible.missing_keys[:4])
        )
    _phase(
        emit,
        "direct_safetensors_load",
        load_started,
        warmer_complete_at_start=warmer_complete_at_start,
        warmer_complete_at_end=bool(warmer_done.is_set()),
    )

    finalize_started = time.monotonic()
    _recreate_rotary_buffers(model, config, Qwen2RotaryEmbedding)
    model.tie_weights()
    _assert_materialized_bf16_model(model)
    _phase(emit, "direct_qwen_finalize", finalize_started)
    return model
