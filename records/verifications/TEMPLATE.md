# Verification Report — NNN-handle

<!-- harness/run_submission.py --save-report generates most of this; the verifier
     completes the human sections. Committed alongside the record's merge. -->

- **Date:**
- **Verifier:** @handle  <!-- mark `self-verified` if verifying your own submission -->
- **PR:** #
- **Hardware:** (nvidia-smi name, driver, VRAM)
- **Harness commit:** · **env.lock sha256:**
- **Target:** ≥ NN.N% GSM8K (spec vN)
- **Seeds:** how chosen (fresh at verification time)

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**Mean train time:** · **vs. current record:**

## Adapter audit

- Trainable params in saved adapter: (cap 30,000,000)
- Loads in pinned peft against frozen base:
- Base model checksum re-verified:

## Code review notes

- Training data sources verified train-split-only:
- Technique summary (what it actually does):
- Concerns / gray areas:

## Verdict

**ACCEPT — new record #N / ACCEPT — notable attempt / REJECT**

Reasoning (written for a stranger auditing this later):
