# Verification Report — t2-001-bacon

- **Date:** 2026-07-27
- **Verifier:** Saivineeth147
- **PR:** #13
- **Hardware:** Modal Sandbox · NVIDIA L40S (network-blocked)
- **Harness commit:** (orchestrator checkout) · Python 3.13.12
- **Target:** ≥ 75.5% SQuAD v1.1 EM (spec-t2 v1)
- **Seeds:** fresh, generated at verification time (below)

## Timed runs

| run | seed | train wall-clock | SQuAD EM | pass |
|-----|------|------------------|-----------|------|
| 1 | 1867645044 | 55.6s | 0.7741721854304636 | ✅ |
| 2 | 318136251 | 48.1s | 0.7672658467360454 | ✅ |
| 3 | 298993731 | 44.9s | 0.7754966887417218 | ✅ |

**Official train time (all-pass, mean of runs): 49.5s** ·
runs passed: 3/3

## Adapter audit

- Trainable params in saved adapter: **14,155,776** (cap 30,000,000) —
  within cap ✅
- Loads in pinned peft against frozen base: ✅ — all three eval runs loaded the saved
  adapter with pinned `peft` against the frozen base and scored, so it is a standard
  loadable PEFT LoRA (the training-time projection fusion is reverted before save).
  Param count identical across all three seeds (14,155,776 = 6 targets × 24 layers, r=16).

## Code review notes

- **Training data sources verified train-split-only:** ✅ `train.py` reads only the
  harness-supplied `--data-dir` (the SQuAD `train` split), takes the 4,000 shortest
  examples by tokenized prompt+answer length, and never opens another path. No reference
  to the `validation` split exists anywhere in the submission. The base model is resolved
  from the **exact pinned snapshot** (`BASE_MODEL_REVISION`), raising if it is absent,
  with no network fallback. No network calls, no subprocesses, no writes outside
  `--output-dir`.
- **Prompt format matches the frozen eval protocol exactly** —
  `Context: {context}\nQuestion: {question}\nAnswer:` — so training and evaluation agree;
  the accuracy is not an artifact of a bespoke prompt.
- **Technique summary:** A faithful port of Track 1 record #4 (004-slippylolo) to
  SmolLM2-1.7B/SQuAD. Six-target BF16 LoRA (q/k/v/o/gate/up, r=16, `down_proj` dropped),
  deterministic best-fit repacking into 1,024-token blocks, per-layer fusion of the frozen
  base and shared-input LoRA-A projections into one GEMM per group, chunked completion-only
  cross-entropy that never materializes full-vocabulary logits, and a 0.75-epoch schedule
  at 5e-4. Combined with shortest-4k pruning this replaces the baseline's 20,000-example
  full-sequence epoch, which is where the bulk of the 11× comes from.
- **Concerns:** None affecting the result. Three parts of the PR were **excluded** from the
  merge as out of scope, and the verification above was run **without** them:
  1. `harness/evaluate_gsm8k.py` / `harness/evaluate_squad.py` were modified to add
     `local_files_only=True` plus a snapshot resolver. The change is unnecessary — the
     harness already exports `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, and these three
     runs evaluated cleanly against the unmodified evaluator. It is also harmful: it
     resolves the model as `sorted(snapshots)[-1]`, i.e. whichever snapshot sorts last,
     while `harness/integrity_check.py` deliberately pins `pins["base_model_sha"]`. Merging
     it would silently weaken the guarantee that every submission is scored against
     identical weights.
  2. A hand-written `records/verifications/t2-001-bacon.md`. Reports are harness-generated
     maintainer artifacts; this file is the one produced by the verification run.
  3. A hand-edited README leaderboard, which is generated from `records/records.json` and
     would fail the CI drift check. It was regenerated instead.
  The submitter also disclosed heavy AI assistance in developing the submission. That is
  not disqualifying — the rules constrain what the code may do, not how it was written —
  and the code is disciplined, notably in resolving the pinned snapshot correctly (which
  the harness patch above did not).

## Verdict

**RECORD (all 3/3 runs passed).** Official mean training time **49.5s** (55.6 / 48.1 /
44.9), against the previous Track 2 record of 525.4s — a **−91%** cut, far outside
run-to-run spread, so this is a clear record rather than a tie. All three fresh seeds clear
the 75.5% target (77.42% / 76.73% / 77.55%), and the adapter is 14,155,776 params, well
under the 30M cap. Code review found the technique legitimate and the data discipline
sound. Self-reported 47.7s vs. the official 49.5s is the expected difference: the harness
clock wraps the entire training process, and the official number is the mean of three runs
rather than a single one. Track 2's first outside submission, and the port shows the Track 1
techniques generalize across model and task rather than being GSM8K-specific.
