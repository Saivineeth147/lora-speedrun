"""Immutable controls for the selected public Track 1 candidate."""

from __future__ import annotations

from types import MappingProxyType


LEARNING_RATE = 8e-4
EPOCH_FRACTION = 0.50
BATCH_SIZE = 8
PACK_LENGTH = 1024
WARMUP_STEPS = 4
MIN_LR_FRACTION = 0.05
LORA_RANK = 16
LORA_ALPHA = 32
ADAM_BETA2 = 0.95
STRIP_CALCULATOR_ANNOTATIONS = True
TRAIN_SUBSET = "shortest:4000"
CE_CHUNK_SIZE = 2048
STARTUP_LOADER = "direct"
PAGE_WARMER = "on"
PREFIX_CACHE_BATCH = "all"
TAIL_SCHEDULE_NAME = "full8-top4"
FULL_BACKWARD_UPDATES = 8
TAIL_UPDATES = 4
BACKWARD_PLAN_NAME = "top1_after8"
MAX_OPTIMIZER_STEPS = FULL_BACKWARD_UPDATES + TAIL_UPDATES
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
)
EXPECTED_SAVED_ADAPTER_PARAMETERS = 13_762_560

PRODUCTION_FEATURES = MappingProxyType(
    {
        "target_down_proj": False,
        "packing": "best_fit",
        "contiguous_hotloop": True,
        "qkv_base_fusion": True,
        "gate_up_base_fusion": True,
        "qkv_lora_a_fusion": True,
        "gate_up_lora_a_fusion": True,
        "adapter_dtype": "bfloat16",
        "backward_plan_source": "fixed_staged_schedule",
        "startup_loader": STARTUP_LOADER,
        "page_warmer": PAGE_WARMER,
        "prefix_cache_build_batch_blocks": PREFIX_CACHE_BATCH,
        "tail_schedule": TAIL_SCHEDULE_NAME,
        "full_backward_updates": FULL_BACKWARD_UPDATES,
        "top_one_backward_updates": TAIL_UPDATES,
    }
)


def projection_components_for(features) -> tuple[str, ...]:
    """Return the sole static projection implementation after an integrity check."""

    if features is not PRODUCTION_FEATURES:
        raise RuntimeError("production feature map changed unexpectedly")
    return ("all_combined",)
