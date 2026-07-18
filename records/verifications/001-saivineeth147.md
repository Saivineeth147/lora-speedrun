# Verification Report — 001-saivineeth147 (Record #1)

- **Date:** 2026-07-18
- **Verifier:** @Saivineeth147 — `self-verified` (per JUDGING.md, maintainer submissions
  are self-verified until a second verifier joins the project)
- **PR:** seed record (established alongside the baseline at launch)
- **Hardware:** Modal Sandbox · NVIDIA L40S (48 GB), network-blocked · Python 3.13.12
- **Harness:** `harness/modal_verify.py` · env pinned via `env.lock` · artifacts pinned
  via `harness/pins.json`
- **Target:** ≥ 57.0% GSM8K (spec v1)
- **Seeds:** fresh, generated at verification time: 463953844, 1272526778, 1645798532

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | 463953844 | 359.8s | 61.64% | ✅ |
| 2 | 1272526778 | 365.2s | 60.65% | ✅ |
| 3 | 1645798532 | 369.7s | 61.11% | ✅ |

**Mean train time: 364.9s (6m 05s)** · runs passed: 3/3 · accuracy 60.6–61.6%
(all clear the 57.0% target by ≥3.6pt).

**vs. current record (#0, 716.9s):** −49% wall-clock, +1.6pt mean accuracy.

## Adapter audit

- Trainable params in saved adapter: **18,464,768** (cap 30,000,000) — within cap ✅.
  Identical to the baseline's adapter — the speedup comes entirely from the data
  pipeline (packing + masking), not from a larger or different adapter.
- Loads in pinned `peft` against the frozen base: ✅.
- Base model + train/test data content hashes re-verified against `harness/pins.json` in
  a fresh eval sandbox before every run: **INTEGRITY: OK** on all 3.

## Code review notes

- **Training data:** GSM8K `train` split only. The `encode` step reads solely
  `ex["question"]`/`ex["answer"]` from the harness-supplied train directory; packing
  concatenates those encodings and nothing else. No test-split reference, no network
  (runs in a `block_network=True` sandbox).
- **Technique:** two changes vs. #0, LoRA config otherwise identical. (1) Completion-only
  loss masking — the `Question: …\nAnswer:` prompt is set to label `-100`, loss computed
  only on the answer completion. (2) Greedy sequence packing into dense 1024-token blocks
  (7,473 examples → 1,510 blocks, ~5 per block), eliminating the baseline's ~80% padding
  waste. Dropped from 3 → 2 epochs.
- **Concerns:** none material. Packing uses full attention across block boundaries (naive
  packing), disclosed in the submission's NOTES.md; negligible for short independent GSM8K
  problems and a standard fast-trainer practice. A block-diagonal mask would remove even
  that and is a legitimate next-record lane. Transient CUDA allocator OOM *warnings*
  appeared at step ~2–4 on all three runs and PyTorch recovered each time (freed cache,
  no fatal error, all completed well within the time limit) — worth a future
  memory-headroom guard, not a correctness issue.

## Verdict

**ACCEPT — new Record #1.**

Beats the current record on both axes: ~2× faster (364.9s vs 716.9s) at higher accuracy
(mean 61.1% vs 59.4%), reproduced across 3 fresh seeds with tight spread (360–370s). The
improvement is fully explained and attributable to two well-understood techniques, and the
adapter audit confirms it isn't smuggling extra capacity. This is exactly the kind of
mechanism-first, reproducible gain the benchmark is built to surface. **New record: 6m 05s.**
