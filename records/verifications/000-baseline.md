# Verification Report — 000-baseline (Record #0)

- **Date:** 2026-07-18
- **Verifier:** @Saivineeth147 — `self-verified` (baseline; per JUDGING.md, maintainer
  submissions are self-verified until a second verifier joins the project)
- **PR:** seed commit (baseline established at track freeze, tag `spec-v1-frozen`)
- **Hardware:** Modal Sandbox · NVIDIA L40S (48 GB), network-blocked · Python 3.13.12
- **Harness:** `harness/modal_verify.py` · env pinned via `env.lock` · artifacts pinned
  via `harness/pins.json`
- **Target:** ≥ 57.0% GSM8K (spec v1)
- **Seeds:** fresh, generated at verification time: 800001841, 1364826184, 1179115513

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | 800001841 | 727.6s | 60.05% | ✅ |
| 2 | 1364826184 | 695.9s | 59.51% | ✅ |
| 3 | 1179115513 | 727.2s | 58.61% | ✅ |

**Mean train time: 716.9s (11m 57s)** · runs passed: 3/3 · accuracy spread 58.6–60.0%
(all clear the 57.0% target with ≥1.6pt margin).

## Adapter audit

- Trainable params in saved adapter: **18,464,768** (cap 30,000,000) — within cap ✅
  (identical across all 3 runs; deterministic given the fixed LoRA config).
- Loads in pinned `peft` against the frozen base: ✅ (eval loads base + adapter each run).
- Base model + train/test data content hashes re-verified against `harness/pins.json` in
  a fresh eval sandbox before every run: **INTEGRITY: OK** on all 3.

## Code review notes

- **Training data:** GSM8K `train` split only, supplied by the harness as an isolated
  `load_from_disk` directory. `train.py` reads nothing else — no other datasets, no test
  split, no network (runs in a `block_network=True` sandbox).
- **Technique:** textbook LoRA SFT. Rank 16, α 32, applied to all 7 linear projections
  (q/k/v/o + gate/up/down), 3 epochs, effective batch 128, cosine LR 2e-4, bf16. No
  packing, no completion-only masking, no quantization, no custom kernels.
- **Concerns:** none material. Run 3 emitted transient CUDA allocator OOM *warnings* that
  PyTorch recovered from by freeing cache; no fatal error, completed within the time
  limit. Noted for future submitters: on this model the batch×seq×vocab logits tensor is
  the memory bottleneck, not the adapter.

## Verdict

**ACCEPT — Record #0 (baseline).**

This establishes the reference point for track v1. It is deliberately naive — its purpose
is to prove the task is achievable and to calibrate the frozen target (set to 57.0%, ~2pts
below the observed accuracy floor to leave seed-variance headroom). Reproducibility is
strong: 3/3 fresh seeds pass, times cluster within ~5% (696–728s), accuracy within 1.5pts.
Every documented weakness in the submission's NOTES.md (no sequence packing, no loss
masking, full-dataset 3-epoch training) is an open lane for the first challenger.
**The record to beat is 11m 57s.**
