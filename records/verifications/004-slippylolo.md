# Verification Report — 004-slippylolo

- **Date:** 2026-07-25
- **Verifier:** Saivineeth147
- **PR:** #12
- **Hardware:** Modal Sandbox · NVIDIA L40S (network-blocked)
- **Harness commit:** (orchestrator checkout) · Python 3.13.12
- **Target:** ≥ 57.0% GSM8K (spec v1)
- **Seeds:** fresh, generated at verification time (below)

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | 432751258 | 41.7s | 0.5890826383623957 | ✅ |
| 2 | 1600440612 | 44.8s | 0.5898407884761183 | ✅ |
| 3 | 1359428397 | 45.2s | 0.6080363912054587 | ✅ |

**Official train time (all-pass, mean of runs): 43.9s** ·
runs passed: 3/3

## Adapter audit

- Trainable params in saved adapter: **13,762,560** (cap 30,000,000) —
  within cap ✅
- Loads in pinned peft against frozen base: ✅ — all three eval runs loaded the saved
  adapter with pinned `peft` against the frozen base and produced scores, so the adapter
  is a standard, loadable PEFT LoRA (the training-time projection fusion is reverted
  before save). Param count is identical across all three seeds (13,762,560).

## Code review notes

- **Training data sources verified train-split-only:** ✅ `train.py` reads only
  `--data-dir` (the harness hands it `gsm8k_train`), takes the 4,000 shortest examples,
  and never opens any other path. The base model is resolved from the pinned local
  snapshot with `local_files_only=True` and **no network fallback** (raises if the pin is
  absent); it also re-checks `harness/pins.json` matches the expected model + SHA. No path
  can reach the test split.
- **Technique summary:** Builds directly on record #3. Six-target LoRA
  (q/k/v/o/gate/up; `down_proj` dropped → 13.76M params, adapter kept in BF16 rather than
  PEFT's FP32 autocast). It (1) deterministically best-fit *repacks the exact next-fit
  membership* of #3 into fewer full 1024-token blocks, (2) fuses each layer's frozen base
  **and** shared-input LoRA-A projections — q/k/v into one GEMM, gate/up into one GEMM —
  (3) computes a chunked completion-only cross-entropy as a custom autograd Function that
  never materializes the full 151k-vocab logits, and (4) shortens the schedule to 0.75
  epoch at 5e-4 with 6 warmup steps.
- **Concerns:** None that block. The projection fusion physically repacks base+LoRA-A
  weights during training, which reads as risky — but it is numerically identical to
  standard PEFT (q/k/v and gate/up each share one input; outputs are split back per
  module), every mutation is validated and fail-closed (a state-dict hook even refuses to
  serialize while packed), and it is fully reversed via `materialize_standard_peft()`
  before save. The saved adapter is therefore an ordinary PEFT LoRA (confirmed: it loaded
  and scored in the fresh eval sandbox). Adapter param count (13,762,560) was recomputed
  by hand from 6 targets × 28 layers at rank 16 and matches exactly. No network, no
  subprocess, writes only under `--output-dir`. Packing-with-no-intra-block-mask is the
  same accepted methodology as #3, and the 3/3 accuracy pass confirms it holds.
- **Infra note:** This is the first sub-minute training run on the leaderboard, and it
  exposed a latent train→eval **volume-commit race** in `harness/modal_verify.py`: the
  fresh eval sandbox could read the output dir before Modal's background commit of the
  train sandbox's adapter write had landed (every prior record trained slowly enough to
  hide it). Fixed by having the orchestrator poll the committed volume (`listdir`) until
  the adapter is visible before launching eval — this runs entirely after the clock stops,
  so timings are unaffected. All three runs above were produced with the fixed harness.

## Verdict

**RECORD (all 3/3 runs passed).** Official mean training time **43.9s** (41.7 / 44.8 /
45.2), a **−58%** improvement over the current record #3 (104.0s) — far outside the
run-to-run spread, so this is a clear record rather than a tie. All three fresh seeds
clear the 57.0% target with margin (58.91% / 58.98% / 60.80%), the adapter is 13,762,560
params (well under the 30M cap) and loads as a standard PEFT LoRA, and code review found
the technique legitimate: a compounding of a shorter 0.75-epoch schedule, a lighter
six-target BF16 adapter, best-fit repacking into fewer blocks, and per-layer base+LoRA-A
GEMM fusion — each a real speedup, none touching the eval path or the timer. The
1m57s→0m44s progression now spans five records and three authors.
