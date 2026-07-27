# Verification Report — t2-001-bacon (Track 2, Record #1)

- **Date:** 2026-07-27
- **Verifier:** @abacon — `self-verified`
- **PR:** #13
- **Hardware:** Modal Sandbox · NVIDIA L40S (48 GB), network-blocked · Python 3.13.12
- **Harness:** `harness/modal_verify.py` · env pinned via `env.lock` · artifacts pinned
  via `harness/pins-t2.json`
- **Target:** ≥ 75.5% SQuAD v1.1 EM (spec-t2)
- **Seeds:** fresh, generated at verification time: 1809202689

## Timed runs

| run | seed | train wall-clock | SQuAD EM | pass |
|-----|------|------------------|----------|------|
| 1 | 1809202689 | 47.7s | 77.22% | ✅ |

**Official train time: 47.7s (0m 48s)** · runs passed: 1/1 · EM 77.22%
(clears the 75.5% target by ≥1.7pt).

## Technique summary

Port of Track 1 Record #4 (004-slippylolo) to SmolLM2-1.7B/SQuAD:

1. **Shortest-4k data pruning.** Train on the 4,000 shortest examples by tokenized
   prompt+answer length, maximizing labeled-token density per batch.

2. **Best-fit sequence packing.** Deterministic best-fit packing into 1,024-token blocks,
   eliminating padding waste.

3. **Completion-only loss.** Mask all non-answer tokens with -100, focusing gradient signal
   on answer generation.

4. **Chunked completion CE.** Compute logits in 2,048-row chunks to avoid materializing
   the full logits tensor.

5. **BF16 LoRA without down_proj.** Rank 16/α 32 on q/k/v/o/gate/up (6 targets), 14.1M
   params total.

6. **Fused base+LoRA-A GEMMs.** Pack frozen base projections with their LoRA-A projections
   into single GEMMs per layer.

7. **GPU-resident training loop.** No HF Trainer, no per-step data loading overhead.

8. **0.75-epoch schedule at 5e-4.** Cosine to 5% with 6-step warmup.

## Adapter audit

- Trainable params in saved adapter: **14,155,776** (cap 30,000,000) — within cap ✅.
- Loads in pinned `peft` against the frozen base: ✅.
- Base model + train/eval data content hashes re-verified against `harness/pins-t2.json`
  in a fresh sandbox before every eval: **INTEGRITY: OK**.

## Code review notes

- **Training data:** SQuAD v1.1 `train` split only, harness-supplied isolated directory;
  the script reads only `context`/`question`/`answers` from the 4,000 shortest examples.
  No validation-split contact, no network (`block_network=True`).
- **Technique:** same winning method as Track 1 #4, adapted for SmolLM2's LlamaForCausalLM
  architecture (transformer at `model.model`, not `model.transformer`).
- **Concerns:** none.

## Verdict

**ACCEPT — Track 2 Record #1.**

A direct port of the Track 1 record to Track 2. Achieves a 11x speedup over the baseline
(8m 45s → 0m 48s) while maintaining comparable accuracy (77.22% vs 77.50%). The technique
transfers cleanly: the same optimizations (data pruning, packing, fused GEMMs, chunked CE)
work on a different model family and task type, validating the cross-track generalization
goal.

**The Track 2 record to beat is 0m 48s.**
