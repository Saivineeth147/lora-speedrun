# Submission notes — t2-000-baseline

**What this is:** the reference implementation and seed record for **Track 2**
(SQuAD v1.1 extractive QA on SmolLM2-1.7B). Same job as track 1's baseline: prove the
task is achievable, calibrate the frozen target, and be beaten.

**Deliberately naive choices** (each one is an open lane):

- Trains on the **first 20,000** of 87,599 examples in natural order — no selection,
  no dedup (SQuAD has ~5 questions per context; smarter subsets are unexplored).
- **Full-sequence loss** — wastes gradient signal on the long context, which the model
  never needs to generate. Completion-only masking took −49% on track 1; here the
  prompt/completion ratio is far more lopsided, so it should matter even more.
- **No packing**, 640-token rows with real lengths well below that.
- Conservative LR, 1 epoch, no quantization, no kernels, no torch.compile.

**Why track 2 exists:** so techniques must *transfer*. A trick that only helps
Qwen-on-GSM8K stalls here; a real training insight should win on both tracks.

**Reproducibility:** any 24 GB+ card runs it locally; official timing is the Modal L40S
sandbox. Verified numbers and seeds are in the verification report on the leaderboard.
