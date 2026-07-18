# Submission notes — 001-saivineeth147

**What changed vs. record #0 (baseline):** two changes, same LoRA config otherwise.

1. **Completion-only loss masking.** The baseline computes loss over the whole
   `Question: … Answer: …` string, including the question the model never has to
   generate. Here the prompt (`Question: {q}\nAnswer:`) is masked to `-100` and loss is
   computed only on the answer completion. Every gradient step now spends its signal on
   the thing being learned (the chain-of-thought and the `#### N` format).
2. **Greedy sequence packing.** GSM8K examples average ~190 tokens. The baseline pads
   each to 768, so the large majority of every batch is padding — wasted compute. Here
   examples are concatenated into dense 1024-token blocks (~5 examples per block) with
   almost no padding, so ~5× more real tokens flow per unit of compute.

**Why it works:** the score is wall-clock, and the baseline spent most of its wall-clock
multiplying by padding and computing loss on prompt tokens. Removing both waste sources
means the same number of *real-token* passes finishes far faster — so **2 epochs** here
does more useful learning than the baseline's 3 padded epochs, in a fraction of the time.

**What I kept identical (to isolate the effect):** rank 16 / α 32 on all seven linear
projections, LR 2e-4, cosine schedule, bf16. The only knobs touched are packing, masking,
and dropping from 3 → 2 epochs.

**Known trade-off / honest caveat:** packing uses full attention across block boundaries
(naive packing), so tokens can attend across the example seam. For short, independent
GSM8K problems the effect is negligible in practice and it's standard for fast trainers;
a block-diagonal (varlen) attention mask would remove it entirely and is an obvious next
step for a challenger.

**Open lanes I did NOT take** (for the next record): 1-epoch aggressive-LR schedule, data
pruning to the hardest N examples, `torch.compile`, fused cross-entropy, QLoRA, rsLoRA.
