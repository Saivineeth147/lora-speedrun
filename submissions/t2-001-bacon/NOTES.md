# t2-001-bacon — Track 2 port of 004-slippylolo: fused BF16 LoRA for SmolLM2/SQuAD

## What this is

A direct port of Track 1's current record (004-slippylolo) to Track 2. The same
optimizations that cut GSM8K training from 8m 45s to 44s are applied to SQuAD on
SmolLM2-1.7B. The technique is unchanged; only the data pipeline and model-specific
access paths differ.

## What changed from the Track 2 baseline (t2-000-baseline)

1. **Shortest-4,000 subset.** SQuAD has 87,599 examples; the baseline naively uses the
   first 20,000. We sort all examples by tokenized prompt+answer length and train on the
   4,000 shortest. Shorter examples pack more efficiently, reducing wasted padding and
   increasing the labeled-token fraction per block.

2. **Best-fit sequence packing.** The baseline uses no packing (one example per 640-token
   row). We deterministically best-fit pack the 4,000 shortest examples into 1,024-token
   blocks, eliminating padding waste entirely.

3. **Completion-only loss.** The baseline computes full-sequence CE loss, wasting gradient
   signal on context tokens. We mask all non-answer tokens with -100, so the model only
   learns to generate answers. This is especially effective for SQuAD, where context
   tokens dominate the sequence.

4. **Chunked completion CE.** The lm_head logits tensor is never materialized for the
   whole batch. Instead, the custom ChunkedCE autograd function computes logits in 2,048-row
   chunks, saving ~6 GB of fp32 logits traffic per step.

5. **BF16 LoRA without down_proj.** The baseline uses PEFT's default FP32 autocast for
   adapters and targets all 7 linear layers including down_proj. We drop down_proj (the
   smallest-impact projection) and keep LoRA parameters in BF16, reducing trainable
   parameters from ~18M to ~9.4M.

6. **Fused base+LoRA-A GEMMs.** Frozen q/k/v base projections and their shared-input
   LoRA-A projections each execute as one packed GEMM per layer. Same for gate/up. This
   reduces per-layer kernel launch overhead and improves GPU utilization.

7. **GPU-resident training loop.** No HF Trainer, no dataloader, no collation, no
   attention mask construction per step. The seeded block permutation and completion
   positions are materialized once; training batches are contiguous views with one final
   loss sync.

8. **0.75-epoch schedule at 5e-4.** The baseline uses 1 epoch at 2e-4. The shorter
   schedule with higher learning rate converges faster on the pruned subset.

## Model-specific adaptations

- **Data reading:** SQuAD has context/question/answers fields; prompt is
  `Context: {c}\nQuestion: {q}\nAnswer:` (matching the frozen eval protocol).
- **Model path resolution:** Pinned SmolLM2-1.7B snapshot from `harness/pins-t2.json`.
- **Transformer access:** SmolLM2 is a `LlamaForCausalLM` internally, so the transformer is at `model.model` (not `model.transformer`).
- **pad_token:** Set to eos_token if not present (SmolLM2 tokenizer has no pad_token by
  default).

## Self-reported results

No local runs yet (Metal GPU, not CUDA). Official verification on Modal L40S pending.

## Methodology and reproducibility

Development uses only the SQuAD TRAIN split. No test examples or predictions informed
candidate selection. The script consumes the verifier-supplied seed, uses only the
provided training directory, resolves the exact pinned SmolLM2 snapshot offline, and has
no network fallback.

No extra packages are required.
