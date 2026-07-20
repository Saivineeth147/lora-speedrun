# Verification Report — t2-000-baseline (Track 2, Record #0)

- **Date:** 2026-07-20
- **Verifier:** @Saivineeth147 — `self-verified` (baseline; per JUDGING.md, maintainer
  submissions are self-verified until a second verifier joins the project)
- **PR:** seed record (Track 2 established at its freeze, tag `spec-t2-frozen`)
- **Hardware:** Modal Sandbox · NVIDIA L40S (48 GB), network-blocked · Python 3.13.12
- **Harness:** `harness/modal_verify.py` · env pinned via `env.lock` · artifacts pinned
  via `harness/pins-t2.json`
- **Target:** ≥ 75.5% SQuAD v1.1 EM (spec-t2)
- **Seeds:** fresh, generated at verification time: 83334717, 1182902862, 350590137

## Timed runs

| run | seed | train wall-clock | SQuAD EM | pass |
|-----|------|------------------|----------|------|
| 1 | 83334717 | 523.5s | 77.60% | ✅ |
| 2 | 1182902862 | 527.2s | 77.62% | ✅ |
| 3 | 350590137 | 953.4s | 77.38% | ✅ |

**Mean train time: 668.0s (11m 08s)** · runs passed: 3/3 · EM spread 77.4–77.6%
(all clear the 75.5% target by ≥1.9pt).

## Variance disclosure (run 3)

Run 3's wall-clock (953.4s) is ~80% above runs 1–2 (~525s). The training loop itself was
normal — 157 steps at ~3.5s/it in all three runs — but run 3's container spent ~6 extra
minutes in pre-training I/O (model/dataset load from the cache volume): platform storage
variance, not anything in the submission. Per the frozen rules, wall-clock includes
loading, so the run counts as-is and the official mean is 668.0s. This makes the t2
baseline slightly *easier* to beat, which is acceptable for a seed record. Logged
publicly as a protocol observation for any future track-spec revision (e.g. median-of-3);
the current rule stands for all v1 tracks.

## Adapter audit

- Trainable params in saved adapter: **18,087,936** (cap 30,000,000) — within cap ✅,
  identical across runs.
- Loads in pinned `peft` against the frozen base: ✅.
- Base model + train/eval data content hashes re-verified against `harness/pins-t2.json`
  in a fresh sandbox before every eval: **INTEGRITY: OK** on all 3.

## Code review notes

- **Training data:** SQuAD v1.1 `train` split only, harness-supplied isolated directory;
  the script reads only `context`/`question`/`answers` from the first 20,000 examples
  (documented naive subset — any subset is legal). No validation-split contact, no
  network (`block_network=True`).
- **Technique:** textbook LoRA SFT, rank 16/α 32 on all 7 linear projections, 1 epoch,
  effective batch 128, cosine LR 2e-4, bf16, full-sequence loss (deliberately naive).
- **Concerns:** none beyond the run-3 I/O variance disclosed above.

## Verdict

**ACCEPT — Track 2 Record #0 (baseline).**

Establishes Track 2's reference point with the same protocol as Track 1: naive baseline,
target frozen at observed − ~2pts (77.50% exploratory → 75.5%). Accuracy is remarkably
stable across seeds (0.24pt spread). The documented naive choices — full-sequence loss
over a long context (far more wasteful here than on Track 1), no packing, arbitrary
first-20k subset — are open lanes. **The Track 2 record to beat is 11m 08s.**
