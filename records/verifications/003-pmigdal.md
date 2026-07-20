# Verification Report — 003-pmigdal (Record #3)

- **Date:** 2026-07-20
- **Verifier:** @Saivineeth147 (independent maintainer verification)
- **PR:** #11
- **Hardware:** Modal Sandbox · NVIDIA L40S (48 GB), network-blocked · Python 3.12.13
- **Harness:** `/verify` automation → `harness/modal_verify.py` · env pinned via `env.lock`
  · artifacts pinned via `harness/pins.json`
- **Target:** ≥ 57.0% GSM8K (spec v1)
- **Seeds:** fresh, generated at verification time: 1618931918, 2048232357, 542430346

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | 1618931918 | 105.3s | 60.88% | ✅ |
| 2 | 2048232357 | 103.4s | 58.30% | ✅ |
| 3 | 542430346 | 103.4s | 61.71% | ✅ |

**Mean train time: 104.0s (1m 44s)** · runs passed: 3/3 · min margin over target +1.30pt.

**vs. record #2 (113.0s): −8%. vs. record #1 (364.9s): −71.5%.** Times are extremely
tight (103.4–105.3s); the author's self-report (98.5s) was within platform noise of the
official mean.

## Adapter audit

- Trainable params: **18,464,768** (cap 30,000,000) — within cap ✅ (script also
  asserts the cap itself).
- Standard PEFT adapter via `save_pretrained` ✅.
- Base model + data content hashes re-verified in fresh sandboxes: **INTEGRITY: OK** ×3.

## Code review notes

- **Data:** harness train dir only; "shortest-4,000 by tokenized length" subset and
  stripping GSM8K `<<...>>` calculator annotations are both within the subset/formatting
  rules. Iteration methodology used a 500-example validation set carved from the END of
  the train split — no test contact during development, exactly per the rules.
- **Technique:** five compounding cuts — shortest-4k pruning, 1 aggressive-LR epoch,
  annotation stripping, a hand-rolled loop over GPU-resident full 1024-token packed
  blocks (no Trainer/dataloader/mask), and a chunked completion-only cross-entropy
  `autograd.Function` that never materializes the full 151k-vocab logits tensor. The
  chunked CE is the standout: it removes the step's dominant memory-bandwidth cost.
  Fixed costs also attacked (page-cache warming and model load overlapped with
  tokenization) — legal, since the whole process is timed.
- **Env-var iteration knobs** default exactly to the submitted config; the harness sets
  none of them. Verified defaults == config.yaml values.
- **Concerns:** none. Accuracy spread across seeds (58.3–61.7%) is wider than earlier
  records — expected at 1 epoch on 4k examples — but all runs cleared with margin.

## Verdict

**ACCEPT — new Record #3. The record to beat is 1m 44s.**

An 11m57s baseline fell to 1m44s in three days — a 6.9× compounded speedup across
three records and two authors, every step verified and mechanism-documented. The
rejected-variants section of the NOTES (Liger unnecessary at GEMM-bound bs=8, 1.25
epochs overfits the pruned subset, 5k subset dominated) is as valuable as the record
itself. This is the leaderboard working as intended.
