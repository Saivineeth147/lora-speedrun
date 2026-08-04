# 005-slippylolo — direct Qwen loading with an 8+4 staged schedule

## What changed from `004-slippylolo`

This submission keeps the six-target BF16 LoRA adapter, shortest-4,000-example
selection, deterministic best-fit packing, fused projection path, contiguous hot
loop, and chunked completion-only cross-entropy from the current Track 1 record.
It changes the startup and training schedule:

1. The pinned Qwen weights are loaded directly from safetensors into a
   meta-constructed model. The Rust tokenizer reads `tokenizer.json` directly.
2. A background sequential read warms the exact model file while Python imports
   and CPU tokenization proceed.
3. Training is capped at 12 optimizer updates: eight full-network backward
   updates followed by four updates through only the top decoder layer.
4. At the transition, the frozen lower 27 layers are evaluated once for all
   remaining packed blocks. Their layer-27 boundary activations are cached in
   BF16 and reused by the four top-layer updates.
5. The first eight learning rates remain unchanged from the 20-update reference
   schedule. Each tail update uses the bounded `8e-4` peak.
6. LoRA injection and export use the direct packed representation while still
   writing a standard two-file PEFT adapter containing all 28 layers.

The full adapter remains active during every forward pass. Staged backward
truncation changes only which LoRA parameters receive gradients after update
eight; it does not prune saved adapter capacity.

## Why it is faster

The direct loader removes high-level model and tokenizer discovery work. Parent
dispatch combines adjacent q/k/v and gate/up base projections and their
shared-input LoRA-A projections. The contiguous hot loop removes dataloader,
collation, attention-mask, and repeated indexing overhead.

The staged tail avoids repeatedly backpropagating through 27 frozen decoder
layers. Batched boundary caching computes those frozen-prefix activations once,
then the final four updates execute only layer 27, the final RMSNorm, and the
completion loss.

## Self-reported results

These are unofficial fixed-candidate measurements on one NVIDIA L40S. The
training wall clock includes model loading, tokenization, optimization, adapter
save, and process exit.

The measured build also generated local aggregate profiling and reproducibility
digests. Those non-training diagnostics are omitted from this public payload;
the schedule, model updates, and adapter export are unchanged, so the timings
reported here are conservative for the submitted source.

| run | seed | train wall-clock | GSM8K exact match |
|---:|---:|---:|---:|
| 1 | `1880479851` | 27.6s | 777/1,319 = 58.91% |
| 2 | `1460648624` | 25.7s | 774/1,319 = 58.68% |
| 3 | `1739596261` | 27.2s | 775/1,319 = 58.76% |

Mean training time: **26.83 seconds**. All three saved adapters exceed the
57.0% target and contain 13,762,560 parameters. Only the maintainer's
fresh-seed verification can establish an official result.

## Reproducibility notes

Development used only the GSM8K training split. The script consumes the
verifier-supplied seed, reads only the provided training directory, resolves
the exact pinned Qwen snapshot offline, and has no network fallback.

The direct loader requires the benchmark's pinned Qwen revision and
Transformers layout. It checks model configuration, tokenizer identity, tensor
inventory, and package version before training. No extra packages are required.
