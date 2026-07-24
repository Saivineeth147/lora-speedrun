"""Pure deterministic packing used by the BF16 best-fit candidate.

The baseline next-fit loop emits a block only when the next example overflows the
current buffer.  It never emits its final buffer.  Membership is therefore frozen by
first reproducing those flushes, not by asking a different packing heuristic which
examples happen to fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class _Example:
    source_index: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PackingResult:
    """Padded blocks plus auditable source-example membership."""

    input_blocks: tuple[tuple[int, ...], ...]
    label_blocks: tuple[tuple[int, ...], ...]
    bin_source_indices: tuple[tuple[int, ...], ...]
    included_source_indices: tuple[int, ...]
    excluded_tail_source_indices: tuple[int, ...]
    baseline_block_count: int
    used_best_fit: bool


def _prepare_examples(
    examples: Sequence[tuple[Sequence[int], Sequence[int]]],
    capacity: int,
) -> tuple[_Example, ...]:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")

    prepared: list[_Example] = []
    for source_index, (raw_ids, raw_labels) in enumerate(examples):
        if len(raw_ids) != len(raw_labels):
            raise ValueError(f"example {source_index} has mismatched ids and labels")
        prepared.append(
            _Example(
                source_index=source_index,
                input_ids=tuple(raw_ids[:capacity]),
                labels=tuple(raw_labels[:capacity]),
            )
        )
    return tuple(prepared)


def _baseline_emitted_bins(
    examples: Sequence[_Example],
    capacity: int,
) -> tuple[tuple[tuple[_Example, ...], ...], tuple[_Example, ...]]:
    """Reproduce baseline next-fit flushes and retain its unflushed final tail."""

    emitted: list[tuple[_Example, ...]] = []
    current: list[_Example] = []
    used = 0
    for example in examples:
        size = len(example.input_ids)
        if used + size > capacity:
            if not current:
                raise AssertionError("a capacity-truncated example cannot overflow an empty bin")
            emitted.append(tuple(current))
            current = []
            used = 0
        current.append(example)
        used += size
    return tuple(emitted), tuple(current)


def _best_fit_bins(
    examples: Sequence[_Example],
    capacity: int,
) -> tuple[tuple[_Example, ...], ...]:
    """Online best-fit in baseline order, with bin creation order as the tie-break."""

    bins: list[list[_Example]] = []
    used: list[int] = []
    for example in examples:
        size = len(example.input_ids)
        choices = [
            (capacity - used[index] - size, index)
            for index in range(len(bins))
            if used[index] + size <= capacity
        ]
        if choices:
            _, bin_index = min(choices)
        else:
            bin_index = len(bins)
            bins.append([])
            used.append(0)
        bins[bin_index].append(example)
        used[bin_index] += size
    return tuple(tuple(bin_examples) for bin_examples in bins)


def best_fit_pack_baseline_membership(
    examples: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    capacity: int,
    pad_token_id: int,
    ignore_index: int = -100,
) -> PackingResult:
    """Best-fit pack exactly the examples emitted by the baseline next-fit loop.

    The baseline's final unflushed buffer is the only excluded set.  A conservative
    non-regression guard falls back to the exact baseline bins if best-fit ever opens
    more bins; either path retains precisely the same example membership.
    """

    prepared = _prepare_examples(examples, capacity)
    baseline_bins, excluded_tail = _baseline_emitted_bins(prepared, capacity)
    included = tuple(example for bin_examples in baseline_bins for example in bin_examples)
    best_fit_bins = _best_fit_bins(included, capacity)

    used_best_fit = len(best_fit_bins) <= len(baseline_bins)
    output_bins = best_fit_bins if used_best_fit else baseline_bins

    input_blocks: list[tuple[int, ...]] = []
    label_blocks: list[tuple[int, ...]] = []
    bin_source_indices: list[tuple[int, ...]] = []
    for bin_examples in output_bins:
        input_ids = tuple(token for example in bin_examples for token in example.input_ids)
        labels = tuple(token for example in bin_examples for token in example.labels)
        if len(input_ids) != len(labels) or len(input_ids) > capacity:
            raise AssertionError("packing produced an invalid bin")
        padding = capacity - len(input_ids)
        input_blocks.append(input_ids + (pad_token_id,) * padding)
        label_blocks.append(labels + (ignore_index,) * padding)
        bin_source_indices.append(tuple(example.source_index for example in bin_examples))

    included_indices = tuple(example.source_index for example in included)
    packed_indices = tuple(index for block in bin_source_indices for index in block)
    if sorted(packed_indices) != sorted(included_indices) or len(packed_indices) != len(
        included_indices
    ):
        raise AssertionError("best-fit changed baseline example membership")
    if len(output_bins) > len(baseline_bins):
        raise AssertionError("best-fit non-regression guard failed")

    return PackingResult(
        input_blocks=tuple(input_blocks),
        label_blocks=tuple(label_blocks),
        bin_source_indices=tuple(bin_source_indices),
        included_source_indices=included_indices,
        excluded_tail_source_indices=tuple(
            example.source_index for example in excluded_tail
        ),
        baseline_block_count=len(baseline_bins),
        used_best_fit=used_best_fit,
    )
