"""Backend-independent proof plan for the contiguous BF16 training hot loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PlannedBatch:
    """One adjacent epoch slice and its completion-only flattened label layout."""

    step: int
    start: int
    stop: int
    source_indices: tuple[int, ...]
    completion_positions: tuple[int, ...]
    completion_targets: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class EpochPlan:
    """The exact seeded order and a complete non-overlapping batch partition."""

    order: tuple[int, ...]
    sequence_length: int
    batches: tuple[PlannedBatch, ...]


def build_epoch_plan(
    order: Sequence[int],
    label_rows: Sequence[Sequence[int]],
    batch_size: int,
    *,
    ignore_index: int = -100,
) -> EpochPlan:
    """Plan exact row permutation, batching, and completion-label flattening.

    Duplicate source indices are valid when an environment override requests more
    than one epoch. Every *selected occurrence* remains present exactly once and in
    the supplied order. Completion positions match
    ``labels[:, 1:].reshape(-1)`` within each batch.
    """

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    frozen_order = tuple(order)
    if not frozen_order:
        raise ValueError("order must contain at least one selected block")
    if not label_rows:
        raise ValueError("label_rows must contain at least one block")
    if any(
        type(index) is not int or not 0 <= index < len(label_rows)
        for index in frozen_order
    ):
        raise ValueError("order contains an invalid source block index")

    sequence_length = len(label_rows[0])
    if sequence_length < 2 or any(len(row) != sequence_length for row in label_rows):
        raise ValueError("label rows must be rectangular with at least two tokens")

    batches: list[PlannedBatch] = []
    flattened_row_width = sequence_length - 1
    for step, start in enumerate(range(0, len(frozen_order), batch_size)):
        stop = min(start + batch_size, len(frozen_order))
        source_indices = frozen_order[start:stop]
        positions: list[int] = []
        targets: list[int] = []
        for relative_row, source_index in enumerate(source_indices):
            for relative_token, target in enumerate(label_rows[source_index][1:]):
                if target != ignore_index:
                    positions.append(
                        relative_row * flattened_row_width + relative_token
                    )
                    targets.append(target)
        batches.append(
            PlannedBatch(
                step=step,
                start=start,
                stop=stop,
                source_indices=source_indices,
                completion_positions=tuple(positions),
                completion_targets=tuple(targets),
            )
        )

    return EpochPlan(
        order=frozen_order,
        sequence_length=sequence_length,
        batches=tuple(batches),
    )
