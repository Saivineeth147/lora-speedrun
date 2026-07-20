# 003-pmigdal — data pruning + 1-epoch LoRA with a lean custom loop

*Vibe-coded with Claude Fable 5 in Claude Code.*

## Technique

Five independent cuts over record #1 (packing + completion-only masking, 2 epochs):

0. **Data pruning: train only on the 4,000 shortest examples** (by tokenized
   prompt+answer length) for exactly 1 epoch. Short GSM8K answers are also
   the cleanest supervision per token; ~484 packed blocks ≈ 61 optimizer steps
   is enough to clear the target with margin.
1. **One epoch, aggressive schedule.** Cosine to 5% with an 8-step warmup and
   AdamW betas (0.9, 0.95) tuned for a ~60-step run.
2. **`<<...>>` calculator annotations stripped** from GSM8K answers (an allowed
   formatting transform) — ~13% fewer training tokens, and the model no longer
   spends capacity learning to emit calculator markup the eval never reads.
3. **No HF Trainer.** Examples are seed-shuffled, greedily packed into *full*
   1024-token blocks (last partial block dropped), and moved to the GPU as one
   tensor before training starts. No dataloader, no collation, no attention mask
   (every block is dense), no per-step host↔device traffic beyond indexing.
4. **Chunked completion-only cross-entropy** (custom `autograd.Function`): the
   lm_head projection is computed only for the ~70% of positions that carry
   labels, in chunks, with d(loss)/d(hidden) formed analytically (softmax − 1)
   inside the forward pass. The full `[batch, 1024, 151936]` logits tensor —
   ~10 GB in fp32 — is never materialized, which removes the single largest
   memory-bandwidth cost of the step and frees VRAM for larger batches.

Fixed-overhead cuts: the model safetensors are read into the page cache on a
background thread while `torch`/`transformers` import, and the model is loaded
to GPU on a second thread while the main thread tokenizes and packs the dataset.
Model resolution goes straight to the cached snapshot (works with or without a
`refs/main` entry in the HF cache).

## Hyperparameters

See `config.yaml`. LoRA r=16 α=32 on all seven linear projections (18,464,768
params), LR 4e-4, batch 8×1024-token blocks, 1 epoch over the 4,000 shortest
examples.

## Methodology (no test-split contact during iteration)

All hyperparameter decisions were made against a 500-example validation set
carved from the END of the train split (training on the remaining 6,973 during
iteration). The official test split was only touched by the unmodified harness
evaluator on trained adapters — the same measurement the verification protocol
performs. Final config was then locked and verified with 3 fresh seeds.

## What was tried and rejected

- **bs=16 or bs=12 blocks**: OOM on the 44 GiB L40S — backward through the frozen
  base saves ~30 GB of activations at bs=8 (down_proj's 8960-dim LoRA input is
  the big one). bs=8 runs at ~7.4k tok/s ≈ 72 effective TFLOPS, already GEMM-bound.
- **LR sweep 3e-4 / 5e-4 / 8e-4 at full data, 1 epoch**: flat (val EM 73.4 / 72.0
  / 73.2) — convergence is not LR-limited, which is what justified pruning data
  instead of tuning the optimizer.
- **1.25 epochs on the shortest-4k subset**: *worse* than 1.0 (val 69.6 vs 70.8) —
  repeating the pruned subset overfits.
- **shortest:5000**: same val EM as 4,000 but +22s — dominated.
- **Liger kernels / torch.compile**: not needed — the step is GEMM-bound at bs=8
  and compile time is on the clock for a ~85s run.

## Self-reported results

Full record-style self-verification on the official Modal L40S harness
(`python harness/modal_verify.py --submission submissions/003-pmigdal --runs 3`,
fresh seeds, network-blocked sandbox, 2026-07-20):

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
| 1 | 511124479 | 102.9s | 59.82% | ✅ |
| 2 | 1675983434 | 95.7s | 58.53% | ✅ |
| 3 | 100133981 | 96.9s | 60.12% | ✅ |

**Mean train time: 98.5s (1m 38.5s)** — vs record #1's 6m 05s (−73%).
Adapter params: 18,464,768 (cap 30,000,000). Verdict: RECORD-ELIGIBLE.
