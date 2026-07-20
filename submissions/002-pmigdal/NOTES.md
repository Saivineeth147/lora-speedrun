# Submission notes — 002-pmigdal

Developed collaboratively with Codex using the GPT-5.6 Sol model.

**What changed vs. record #1:** Training is reduced from two epochs to one, with the
peak learning rate raised from 2e-4 to 4e-4. Packs are at most 512 tokens instead of
1024, and the per-device batch is 8 instead of 4. The LoRA configuration,
completion-only objective, cosine schedule, and gradient accumulation are unchanged.

**Why it works:** Causal self-attention cost grows quadratically with sequence length.
Halving the packed sequence length reduces attention work while doubling the batch keeps
approximately the same number of tokens in each forward/backward pass. It also reduces
memory pressure and the amount of unrelated-example context visible across pack seams.
The MLP still dominates much of this model's compute, so shorter packs alone produced
only a modest gain. The single epoch is what targets a substantial wall-clock reduction;
doubling the peak LR approximately preserves the cosine schedule's integrated learning
rate in half the passes.

**Trade-off:** Examples longer than 512 tokens are truncated, versus the record's
1024-token limit. The script reports how many are affected so the accuracy/time result
can be interpreted honestly.

**Reproducibility notes:** Developed for the frozen Track 1 Modal L40S environment.
The tokenizer and model explicitly select the commit frozen in `harness/pins.json`, so
offline loading does not depend on a mutable `main` cache reference.

**Prior two-epoch ablation (commit `c39d54c`):** One fresh-seed Modal L40S run (seed `680784449`) trained in
349.7 seconds (5:50) and scored 60.65% GSM8K EM (800/1,319). This passes comfortably
and is 15.2 seconds faster than record #1's 364.9-second mean, but the improvement is
only about 4%. We intentionally stopped replication because the result is a useful
ablation, not the substantial (~2x) speedup we want to pursue next.

**One-epoch result:** One fresh-seed Modal L40S run (seed `1127653477`) trained in
208.5 seconds (3:29) and scored 60.65% GSM8K EM (800/1,319). This is 42.9% faster
than record #1's 364.9-second mean while retaining its accuracy. It establishes that
the second epoch was unnecessary when the peak LR is doubled.

**Current experiment:** Train the same one-epoch recipe on the first 6,000 examples
(80.3% of the train split). The one-epoch run has 3.65 points of accuracy margin; this
tests whether some of that margin can be traded for a literal >2x wall-clock speedup.

**6,000-example result:** One fresh-seed Modal L40S run (seed `157552210`) trained
in 186.3 seconds (3:06) and scored 60.27% GSM8K EM (795/1,319). This is 49.0%
faster than record #1, with only a 0.38-point drop from the full one-epoch run.

**Current conceptual experiment:** Halve the data again to 3,000 examples and double
the peak LR to 8e-4. This keeps the cosine schedule's approximate integrated LR near
the successful 6,000-example run while halving token exposure. If capability tracks
integrated update magnitude more strongly than unique-example count in this regime,
the target may survive at roughly 70% less wall-clock than record #1.

**3,000-example result:** One fresh-seed Modal L40S run (seed `1249599444`) trained
in 103.4 seconds (1:43) and scored 59.89% GSM8K EM (790/1,319). This is 71.7%
faster than record #1 while remaining 2.89 points above target. Halving the unique
examples and doubling peak LR cost only 0.38 points versus the 6,000-example run.

**Why that version was rejected:** A replication seed (`184398192`) scored only
57.70% (761/1,319). It technically passed, but that margin is too fragile to pursue.

**Current conceptual experiment:** Keep the same 3,000 examples and token compute, but
halve gradient accumulation (8 -> 4) and peak LR (8e-4 -> 4e-4). This doubles optimizer
updates from 22 to 43 while preserving approximate integrated LR. The hypothesis is that
the 22-update regime was optimization-starved and seed-sensitive, not data-starved.

**43-update exploratory result:** One fresh-seed Modal L40S run (seed `1095429334`)
trained in 102.6 seconds (1:43) and scored 59.59% (786/1,319). It preserves the ~72%
speedup with 2.59 points of target margin. We require fresh-seed replication above an
internal 59% floor before treating it as robust, stricter than the official 57% rule.

**Three-seed result:** The stabilized 43-update recipe reproduced at 102.6s/59.59%,
124.9s/59.67%, and 139.0s/60.42% on seeds `1095429334`, `405987357`, and
`2139737663`. Mean timed training was 122.2 seconds (2:02), 66.5% faster than record
#1's 364.9-second mean. All three clear our internal 59% floor; worst-case margin over
the official target is 2.59 points. Timing variance came from sandbox/model-loading
overhead; the training loop itself remained approximately 68–71 seconds.
