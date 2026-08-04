"""Fixed learning-rate sequence for the public 8+4 staged schedule."""

from __future__ import annotations

import hashlib
import json
import math


REFERENCE_OPTIMIZER_UPDATES = 20
PEAK_LEARNING_RATE = 8e-4
WARMUP_UPDATES = 4
MINIMUM_LR_FRACTION = 0.05
SCHEDULE_NAME = "full8-top4"
FULL_BACKWARD_UPDATES = 8
TAIL_UPDATES = 4


def _reference_lr(step: int) -> float:
    if not 0 <= step < REFERENCE_OPTIMIZER_UPDATES:
        raise ValueError("reference step is outside the 20-update schedule")
    if step < WARMUP_UPDATES:
        return PEAK_LEARNING_RATE * (step + 1) / WARMUP_UPDATES
    progress = (step - WARMUP_UPDATES) / (
        REFERENCE_OPTIMIZER_UPDATES - WARMUP_UPDATES
    )
    return PEAK_LEARNING_RATE * (
        MINIMUM_LR_FRACTION
        + (1.0 - MINIMUM_LR_FRACTION)
        * 0.5
        * (1.0 + math.cos(math.pi * progress))
    )


REFERENCE_SEQUENCE = tuple(
    _reference_lr(step) for step in range(REFERENCE_OPTIMIZER_UPDATES)
)
LEARNING_RATE_SEQUENCE = (
    REFERENCE_SEQUENCE[:FULL_BACKWARD_UPDATES]
    + (PEAK_LEARNING_RATE,) * TAIL_UPDATES
)


def _require_schedule(name: str) -> None:
    if name != SCHEDULE_NAME:
        raise RuntimeError(f"unsupported fixed schedule: {name!r}")


def learning_rate_sequence(name: str) -> tuple[float, ...]:
    _require_schedule(name)
    return LEARNING_RATE_SEQUENCE


def learning_rate_at(name: str, step: int) -> float:
    sequence = learning_rate_sequence(name)
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step < len(sequence)
    ):
        raise ValueError("step is outside the selected fixed schedule")
    return sequence[step]


def full_backward_updates(name: str) -> int:
    _require_schedule(name)
    return FULL_BACKWARD_UPDATES


def schedule_config(name: str) -> dict[str, object]:
    sequence = learning_rate_sequence(name)
    tail = sequence[FULL_BACKWARD_UPDATES:]
    reference_prefix = REFERENCE_SEQUENCE[:FULL_BACKWARD_UPDATES]
    reference_tail = REFERENCE_SEQUENCE[FULL_BACKWARD_UPDATES:]
    encoded = json.dumps(
        sequence,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {
        "schedule_family": "fixed_prefix_peak_capped_tail",
        "schedule_name": name,
        "reference_optimizer_updates": REFERENCE_OPTIMIZER_UPDATES,
        "full_backward_updates": FULL_BACKWARD_UPDATES,
        "top_one_backward_updates": len(tail),
        "total_optimizer_updates": len(sequence),
        "reference_tail_updates": len(reference_tail),
        "prefix_lr_area": sum(reference_prefix),
        "reference_tail_lr_area": sum(reference_tail),
        "actual_tail_lr_area": sum(tail),
        "total_lr_area": sum(sequence),
        "tail_lr_start": tail[0],
        "tail_lr_end": tail[-1],
        "tail_area_ratio": sum(tail) / sum(reference_tail),
        "lr_sequence_sha256": hashlib.sha256(encoded).hexdigest(),
        "optimizer_state_transition_policy": "carry_without_reset",
    }
