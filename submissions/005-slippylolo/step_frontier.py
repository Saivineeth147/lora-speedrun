"""Immutable optimizer-step horizon for the selected Track 1 candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random


MAX_STEPS_LIMIT = 1_000_000


@dataclass(frozen=True)
class TrainingHorizon:
    """One authenticated derivation of blocks, updates, and throughput scope."""

    source_blocks: int
    requested_epoch_fraction: float
    batch_size: int
    uncapped_blocks: int
    uncapped_optimizer_steps: int
    max_optimizer_steps: int | None
    total_blocks: int
    optimizer_steps: int

    @property
    def capped(self) -> bool:
        return self.total_blocks < self.uncapped_blocks

    @property
    def effective_epoch_fraction(self) -> float:
        return self.total_blocks / self.source_blocks

    def processed_blocks_after(self, step: int) -> int:
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not 0 <= step < self.optimizer_steps
        ):
            raise ValueError("step must identify a planned optimizer update")
        return min((step + 1) * self.batch_size, self.total_blocks)

    def to_dict(self) -> dict[str, bool | float | int | None]:
        return {
            "source_blocks": self.source_blocks,
            "requested_epoch_fraction": self.requested_epoch_fraction,
            "effective_epoch_fraction": self.effective_epoch_fraction,
            "batch_size": self.batch_size,
            "uncapped_blocks": self.uncapped_blocks,
            "uncapped_optimizer_steps": self.uncapped_optimizer_steps,
            "max_optimizer_steps": self.max_optimizer_steps,
            "planned_blocks": self.total_blocks,
            "planned_optimizer_steps": self.optimizer_steps,
            "capped": self.capped,
        }


def build_training_horizon(
    source_blocks: int,
    epoch_fraction: float,
    batch_size: int,
    max_steps: int | None,
) -> TrainingHorizon:
    """Cap the natural epoch-derived horizon before any row order is built."""

    if (
        isinstance(source_blocks, bool)
        or not isinstance(source_blocks, int)
        or source_blocks < 1
    ):
        raise ValueError("source_blocks must be a positive integer")
    if (
        isinstance(epoch_fraction, bool)
        or not isinstance(epoch_fraction, (int, float))
        or not math.isfinite(epoch_fraction)
        or epoch_fraction <= 0
    ):
        raise ValueError("epoch_fraction must be a positive finite number")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if max_steps is not None and (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= MAX_STEPS_LIMIT
    ):
        raise ValueError("max_steps must be null or a bounded positive integer")

    uncapped_blocks = round(source_blocks * epoch_fraction)
    if uncapped_blocks < 1:
        raise ValueError("epoch_fraction selects no training blocks")
    uncapped_steps = (uncapped_blocks + batch_size - 1) // batch_size
    optimizer_steps = (
        uncapped_steps
        if max_steps is None
        else min(uncapped_steps, max_steps)
    )
    total_blocks = min(uncapped_blocks, optimizer_steps * batch_size)
    if (
        total_blocks < 1
        or (total_blocks + batch_size - 1) // batch_size != optimizer_steps
    ):
        raise RuntimeError("step cap produced an inconsistent training horizon")

    return TrainingHorizon(
        source_blocks=source_blocks,
        requested_epoch_fraction=float(epoch_fraction),
        batch_size=batch_size,
        uncapped_blocks=uncapped_blocks,
        uncapped_optimizer_steps=uncapped_steps,
        max_optimizer_steps=max_steps,
        total_blocks=total_blocks,
        optimizer_steps=optimizer_steps,
    )


def build_shuffled_block_order(
    horizon: TrainingHorizon,
    rng: Random,
) -> list[int]:
    """Build exactly the capped number of seeded source-block occurrences."""

    if not isinstance(horizon, TrainingHorizon):
        raise TypeError("horizon must be a TrainingHorizon")
    order: list[int] = []
    while len(order) < horizon.total_blocks:
        epoch = list(range(horizon.source_blocks))
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[: horizon.total_blocks]


def cosine_learning_rate(
    step: int,
    horizon: TrainingHorizon,
    *,
    peak: float,
    warmup_steps: int,
    minimum_fraction: float,
) -> float:
    """Evaluate the existing warmup/cosine schedule on the capped horizon."""

    if not isinstance(horizon, TrainingHorizon):
        raise TypeError("horizon must be a TrainingHorizon")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step < horizon.optimizer_steps
    ):
        raise ValueError("step must identify a planned optimizer update")
    if (
        isinstance(peak, bool)
        or not isinstance(peak, (int, float))
        or not math.isfinite(peak)
        or peak <= 0
    ):
        raise ValueError("peak must be a positive finite number")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 1
    ):
        raise ValueError("warmup_steps must be a positive integer")
    if (
        isinstance(minimum_fraction, bool)
        or not isinstance(minimum_fraction, (int, float))
        or not math.isfinite(minimum_fraction)
        or not 0 <= minimum_fraction <= 1
    ):
        raise ValueError("minimum_fraction must be finite and between zero and one")

    if step < warmup_steps:
        return float(peak) * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(
        1,
        horizon.optimizer_steps - warmup_steps,
    )
    return float(peak) * (
        float(minimum_fraction)
        + (1 - float(minimum_fraction))
        * 0.5
        * (1 + math.cos(math.pi * progress))
    )
