# Verification Report — 002-pmigdal (Record #2)

- **Date:** 2026-07-20
- **Verifier:** @Saivineeth147 (independent maintainer verification — first outside submission)
- **PR:** #10
- **Hardware:** Modal Sandbox · NVIDIA L40S (48 GB), network-blocked · Python 3.12.13
- **Harness:** `/verify` automation → `harness/modal_verify.py` · env pinned via `env.lock`
  · artifacts pinned via `harness/pins.json`
- **Target:** ≥ 57.0% GSM8K (spec v1)
- **Seeds:** fresh, generated at verification time: 1230150554, 737600345, 416179408

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | 1230150554 | 112.4s | 58.45% | ✅ |
| 2 | 737600345 | 116.0s | 59.82% | ✅ |
| 3 | 416179408 | 110.5s | 60.42% | ✅ |

**Mean train time: 113.0s (1m 53s)** · runs passed: 3/3 · min margin over target +1.45pt.

**vs. previous record (#1, 364.9s): −69%.** Official mean beat the author's self-report
(122.2s) — self-reported numbers held up.

## Adapter audit

- Trainable params: **18,464,768** (cap 30,000,000) — within cap ✅.
- Standard PEFT adapter, loads against the frozen base ✅.
- Base model + data content hashes re-verified in fresh sandboxes: **INTEGRITY: OK** ×3.

## Code review notes

- **Data:** harness-supplied train dir only; 3,000-example subset selection is within
  the "any subset" rule. No test-split contact; pinned-snapshot model resolution reads
  `harness/pins.json` (read-only — legitimate offline-loading aid).
- **Technique:** one 4e-4 epoch over 3,000 examples with doubled optimizer updates
  (512-token packs, microbatch 8) — holds the cosine schedule's integrated LR while
  halving token exposure. NOTES.md documents the full iteration trail, including a
  *rejected* 3k @ 8e-4 variant whose replication margin was too fragile (57.7%) —
  exactly the self-policing this leaderboard wants to reward.
- **Concerns:** none.

## Verdict

**ACCEPT — new Record #2.**

First outside record on the leaderboard. Beats record #1 by 69% with all three
fresh-seed runs clearing target, and the official rerun landed *faster* than the
self-report. The notes double as an ablation study (2-epoch → 1-epoch → 6k → 3k) that
future submitters can build on. Record to beat: superseded same day by #3 — see the
next report.
