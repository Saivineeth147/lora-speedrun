# 004-slippylolo — fused six-target LoRA with a 0.75-epoch schedule

## What changed from `003-pmigdal`

This submission keeps the shortest-4,000-example, packed, completion-only training
strategy and makes five changes:

1. It removes `down_proj` from the LoRA targets, reducing the saved adapter from
   18,464,768 to 13,762,560 parameters.
2. LoRA parameters remain in BF16 instead of PEFT's default FP32 autocast path.
3. Deterministic best-fit packing repacks exactly the examples emitted by the original
   next-fit loop while producing fewer full blocks when possible.
4. Frozen q/k/v and gate/up projections, together with each group's shared-input LoRA-A
   projections, are fused into one base GEMM and one LoRA-A GEMM per group.
5. The schedule changes from 1.0 epoch at `4e-4` with 8 warmup steps to 0.75 epoch at
   `5e-4` with 6 warmup steps.

The seeded block permutation, completion positions, and completion targets are
materialized once. Training then uses contiguous views with no dataloader, collation,
attention-mask, or per-step host-to-device overhead.

## Self-reported results

These are unofficial fixed-candidate measurements on one NVIDIA L40S. The wall clock
includes model loading, tokenization, training, adapter save, and process exit.

| run | seed | train wall-clock | GSM8K exact match |
|---:|---:|---:|---:|
| 1 | `3580952` | 52.4s | 772/1,319 = 58.53% |
| 2 | `705619208` | 48.4s | 771/1,319 = 58.45% |
| 3 | `1674648533` | 54.0s | 768/1,319 = 58.23% |

Mean training time: **51.6 seconds**. All three saved adapters exceed the 57.0% target
and contain 13,762,560 parameters. Only the maintainer's fresh-seed verification can
establish an official result.

## Methodology and reproducibility

Development used only the GSM8K TRAIN split. No test examples or predictions informed
candidate selection. The script consumes the verifier-supplied seed, uses only the
provided training directory, resolves the exact pinned Qwen snapshot offline, and has
no network fallback.

No extra packages are required. A 4,096-row completion-CE chunk was slower than the
retained 2,048-row chunk. In the fixed three-seed TRAIN-only comparison, the `5e-4`
schedule had the highest aggregate score among the tested shorter-schedule points at
`4.5e-4`, `5e-4`, and `5.25e-4`.
