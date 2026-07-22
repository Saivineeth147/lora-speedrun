# Submission notes — 002-shravandoda

**What changed vs. record #1:** enabled Liger Kernel and increased the per-device
microbatch from 4 to 8 while reducing gradient accumulation from 8 to 4. The effective
batch remains 32 packs. The data pipeline, LoRA configuration, optimizer, schedule, and
two-epoch duration are unchanged.

Liger's fused linear cross-entropy removes the full logits allocation, creating enough
memory headroom for the larger microbatch. Processing twice as many packs per forward and
backward pass improves GPU utilization and halves the number of gradient-accumulation
boundaries without changing the examples represented by each optimizer update. Liger also
provides fused RoPE, RMSNorm, and SwiGLU kernels through Transformers' integration.

One self-verification run on the Modal L40S completed in 321.1 seconds and scored 61.41%
(810/1319) on GSM8K with 18,464,768 adapter parameters. The seed was 42. This beats the
current record's 364.9-second mean in the measured run, but the required three fresh-seed
maintainer verification is still pending.

The submission retains record #1's naive sequence packing behavior: examples in the same
pack can attend across boundaries.
