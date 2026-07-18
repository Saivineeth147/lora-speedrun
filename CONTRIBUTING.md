# Contributing

Two ways in: **submit a record attempt** (the fun one), or improve the harness/docs
(also welcome — harness PRs that could affect timing or scoring get extra scrutiny).

## Submitting a record attempt

1. Read [TASK.md](./TASK.md) — the rules are short and strictly enforced.
2. Copy `submissions/TEMPLATE/` to `submissions/NNN-<your-github-handle>/` where `NNN` is
   the next unused number.
3. Your directory must contain:
   - **`train.py`** — implements the submission contract
     (`--data-dir`, `--output-dir`, `--seed`). Must run standalone on the pinned env.
   - **`config.yaml`** — author info, one-line technique description, every hyperparameter,
     and any extra pinned pip packages you need.
   - **`NOTES.md`** — what you changed vs. the current record and *why it works*
     (mechanism, not vibes). This is the part future readers learn from.
4. Run the verification yourself first on a rented 4090:
   ```bash
   python harness/run_submission.py submissions/NNN-yourhandle --runs 3
   ```
   Put your 3 times/accuracies/seeds in the PR description. Don't commit adapter weights.
5. Open a PR titled: `[record] <handle> — <one-line technique> — <mm:ss self-reported>`

No GPU access for a first try? Open the PR with `--runs 1` numbers from any 24 GB card and
say so — if the approach looks plausible, verification will still run on spec hardware.
It just moves slower in the queue.

## What happens next

A maintainer will (usually within 72h):

1. Statically validate your submission (CI does this on the PR automatically).
2. Rerun it 3× with **fresh seeds** on a spec RTX 4090 with the pinned environment.
3. Audit the adapter (param count, base checksum) and review your code against the rules.
4. Post a **public verification report** on the PR: per-run times, accuracies, seeds,
   hardware fingerprint, and an accept/reject verdict with reasoning.

Outcomes:

- **New record** — beats the current record's mean time, all 3 runs pass, rules-clean.
  Merged; leaderboard updated; verification report committed to `records/verifications/`.
- **Notable attempt** — doesn't beat the record (or misses target on 1 of 3 runs) but the
  technique is genuinely novel/instructive. Merged into `submissions/` and listed in
  RECORDS.md's notable-attempts section, without taking the record.
- **Rejected** — doesn't reproduce, breaks a rule, or adds nothing over a merged record.
  Reasoning stays public on the PR.

## Judging rubric

| Check | Pass condition |
|---|---|
| Reproducibility | All 3 fresh-seed reruns clear the target on spec hardware |
| Speed | Mean rerun time beats the current record |
| Rule compliance | No banned data, no test-set contact, adapter within param cap |
| Honesty | Self-reported numbers within noise of verified numbers |
| Explanation | NOTES.md explains the mechanism clearly enough for others to build on |

Full protocol and threat model: [JUDGING.md](./JUDGING.md).

## Harness / docs PRs

Welcome. Anything touching `harness/`, `spec.yaml`, or timing gets re-tested on GPU before
merge, because scoring integrity depends on it. Typos and docs merge fast.

## Questions

Open a GitHub issue or discussion. Rule-interpretation questions get answered publicly and
folded back into TASK.md so the rulebook stays complete.
