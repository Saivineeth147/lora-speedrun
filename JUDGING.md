# Judging & Verification Protocol

Every record on the leaderboard was independently re-run and reviewed by a maintainer
before merge. This file is the complete protocol, so the process is auditable and so
anyone can check that a record earned its place. Nothing on the leaderboard is
self-reported.

## The pipeline

Every record-attempt PR goes through, in order:

1. **Static validation (automated, CI).** `harness/validate_submission.py` — required
   files present, config schema valid, base model matches `spec.yaml`, `train.py` parses,
   no weight files smuggled into the PR.
2. **AI security screen (automated, advisory).** A Claude agent reviews the diff for
   exfiltration attempts, network use, harness/timer tampering, test-set contact, and
   rule violations, and posts its findings as a public PR comment. It gates nothing —
   it exists to make step 3 sharper and its findings part of the public record.
3. **Code review (human).** The verifier reads `train.py` and `config.yaml` in full before
   running anything: data sources (train split only?), any path that could touch the test
   split, any base-weight mutation, timer-adjacent tricks, dependency list sanity.
4. **Timed reruns (sandboxed).** The verifier comments `/verify`, which runs
   `harness/modal_verify.py <dir> --runs 3` with **fresh seeds chosen at verification
   time** (recorded in the report), inside a network-blocked Modal Sandbox on the spec
   L40S with the pinned environment (`env.lock`) and pinned artifacts
   (`harness/pins.json`). Downloads are pre-cached; the harness owns the timer.
   Evaluation happens in a separate fresh sandbox that first re-verifies model and
   dataset content hashes — see SECURITY.md for the full isolation model.
5. **Adapter audit.** The harness counts every parameter in the saved adapter safetensors
   (cap: 30M) — `modules_to_save` full-layer copies count. Adapter must load in pinned
   `peft` against the frozen base.
6. **Verdict.** All 3 runs must clear the target. The official time is the **mean of the
   runs, after dropping any infrastructure-noise outlier** (see *What the official time is*
   below); it takes the record if it beats the current one, otherwise notable-attempt or
   reject per the rubric in CONTRIBUTING.md.
7. **Public report.** The verification report is posted as a PR comment (automatically by
   the `/verify` workflow) and committed to `records/verifications/NNN-handle.md` with the
   verifier's written reasoning. The merge commit links both.

## What the official time is

The ranking quantity gets the same rigor as the accuracy gate — because the number you
protect (accuracy) is not the number that decides the order (time).

- **Mean of the runs, never the fastest.** A record is the mean of the timed runs, not the
  single quickest one, so no lucky-fast run stands on its own.
- **Infrastructure-noise outliers are dropped and re-run.** The timing noise here is
  one-sided: a shared Modal volume can only ever make a run *slower* (a cold-cache I/O
  stall), never faster than the algorithm. So a run whose training wall-clock exceeds
  **1.5× the batch median while its accuracy is normal** is treated as infrastructure noise,
  not algorithmic time — its time is dropped from the mean and the run is re-run. The rule
  only ever drops an accuracy-passing run and never leaves fewer than two timed runs, so a
  genuinely slow submission is never hidden. (This is also why the statistic is the mean and
  not the raw median: with only three runs the median can discard the *cleanest* run, while
  the outlier rule removes only the contaminated one.) Implemented in
  [`harness/run_submission.py`](./harness/run_submission.py) (`official_train_time`).
- **One verification per code state.** Fresh seeds per verification stop seed-shopping on
  accuracy — but re-requesting `/verify` until a batch comes back lucky-fast would just move
  that cherry-pick onto the clock. So a submission is verified once per code state:
  re-verify only after a code change, and the verifier draws fresh seeds each time.

## Threat model → countermeasure

| Attack | Countermeasure |
|---|---|
| Seed shopping (works on one lucky seed) | Verifier picks 3 fresh seeds at rerun time; all must pass |
| Batch resampling (re-request `/verify` until a lucky-fast batch) | One verification per code state; verifier draws fresh seeds each time; re-verify only after a code change |
| Datacenter I/O inflating the timed mean | A run >1.5× the batch median at normal accuracy is dropped from the official time and re-run — see *What the official time is* |
| Training on the test split | Harness hands `train.py` a directory containing *only* the train split; code review; leakage heuristics (suspiciously high accuracy vs. training time gets extra scrutiny, incl. n-gram overlap spot checks) |
| "Adapter" that is actually a full fine-tune | Every tensor in the adapter file counts against the 30M cap |
| Malicious submission code (exfiltration, mining, persistence) | Runs only in a network-blocked Modal sandbox with zero secrets — see SECURITY.md |
| Poisoning the shared cache (base model, datasets) during training | Eval runs in a separate fresh sandbox that re-hashes model + train/test data against `harness/pins.json` first |
| Stealing CI tokens via a booby-trapped PR | Workflows never execute PR code; `/verify` grafts only `submissions/` onto trusted main and refuses PRs touching anything else |
| Timer games (deferring work to eval, background pre-warm) | Harness owns the clock around the whole training process; fresh eval sandbox; eval protocol is fixed and identical for everyone; code review |
| Env games (secret faster wheels) | Verification always uses `env.lock`; extra deps must be declared, pinned, and open-source |
| Prompt-injecting the AI screener | The screener is advisory-only and instructed to flag injection attempts; nothing executes without a human `/verify`, and execution is sandboxed regardless |
| Self-reported numbers that don't hold up | Only verified numbers ever reach the leaderboard |

## Verifier integrity

- The **maintainer's own submissions** (including the baseline) go through the identical
  pipeline, and their verification reports are marked `self-verified`. As soon as the
  project has a second verifier, maintainer submissions must be verified by someone else.
  Want to be a verifier? You need reliable access to a spec 4090 and a track record of
  submissions; open an issue.
- Verification reports include the hardware fingerprint, harness git commit, env.lock
  hash, and all seeds — and because official hardware is a Modal L40S covered by Modal's
  free monthly credits, **anyone can independently re-verify any record for free** with
  `python harness/modal_verify.py --submission <dir> --runs 3 --seeds <from-the-report>`.
  Disputes: open an issue titled `[dispute] record #N`; the record gets re-run publicly.

## Report format

Reports follow [`records/verifications/TEMPLATE.md`](./records/verifications/TEMPLATE.md):
per-run table (seed, wall-clock, accuracy, pass), adapter audit, code-review notes, and the
verdict with reasoning. Write the reasoning so a stranger can follow why the call was made —
the reports double as the project's technical changelog.
