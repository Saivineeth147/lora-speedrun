# Submission notes — 000-baseline

**What this is:** the reference implementation and record #0. Plain LoRA supervised
fine-tuning with textbook-default hyperparameters. It exists to (a) prove the task is
achievable, (b) calibrate the frozen target, and (c) be beaten.

**What it deliberately does NOT do** (free ideas, in roughly increasing effort):

- No completion-only loss masking — it trains on the question tokens too, wasting loss
  signal on text the model never needs to generate.
- No sequence packing — GSM8K examples average ~150 tokens, so most of every 768-token
  batch row is padding. Packing alone should be a large speedup.
- No data selection — all 7,473 examples, 3 full epochs, natural order. Almost certainly
  far more compute than needed to clear the bar.
- No quantization, no custom kernels, no `torch.compile`, no fused cross-entropy.
- Conservative LR (2e-4) and a long schedule. Short aggressive schedules are unexplored.

**Reproducibility:** any 24 GB card runs this comfortably; timing on non-4090 hardware
is not comparable. Verified numbers and seeds are in the verification report linked from
the leaderboard.
