# Security Model

This project executes **arbitrary code from strangers** (submission `train.py` files).
This document is the complete model of how that's made safe. If you find a hole in it,
open a private security advisory on GitHub — good-faith reports are credited.

## Where untrusted code runs (and where it never runs)

Submission code executes in exactly one place: a **Modal Sandbox** — an isolated
container on Modal's infrastructure with:

- **`block_network=True`** — no outbound network. It can't exfiltrate anything, phone
  home, join a mining pool, or download extra payloads.
- **No secrets** — the sandbox receives zero environment credentials. There is nothing
  to steal: the task needs no keys at all (ungated model + dataset, offline runs).
- **A hard timeout** — runaway or stalling code is killed; cost exposure is bounded
  (also set a spend limit in the Modal dashboard).

Untrusted code **never** runs on: a maintainer's personal machine, the GitHub Actions
runner, or any environment holding a token.

## Where the credentials live

| Credential | Lives in | Ever near submission code? |
|---|---|---|
| Modal token | GitHub Actions secrets + maintainer's own machine | Never — orchestrator side only |
| Claude OAuth token (reviewer agent) | GitHub Actions secrets | Never — agent reads the diff as text via the GitHub API |
| GitHub token | Ephemeral per-workflow | Never — no workflow executes PR code |

Fork PRs cannot read repository secrets (GitHub's design). The two workflows that *do*
hold secrets never execute PR-supplied code:

- `verify.yml` checks out **trusted main**, then grafts in *only* the `submissions/`
  directory from the PR head as data files, and **refuses to run at all** if the PR
  touches anything else (harness, workflows, spec). Only maintainers can trigger it.
- `agent-review.yml` checks out trusted main and reads the PR exclusively as diff text
  through the GitHub API.

Changes to `harness/`, `.github/`, or `spec.yaml` are never auto-verified — they get
human review, on the explicit assumption that their author is trying to steal the
tokens or rig the timer.

## Cache-poisoning defense

Training and evaluation share a cache volume (base model, datasets), and training code
can write to it. So evaluation runs in a **separate, fresh sandbox** that first re-hashes
the base model files and the train/test data against committed pins
(`harness/pins.json`, enforced by `harness/integrity_check.py`). A run that tampers
with the cache doesn't get a score — it gets an integrity failure and a public reject.

The eval sandbox is built from a clean image, so `sitecustomize.py` tricks, poisoned
site-packages, or modified harness files from the training sandbox don't survive into
evaluation either.

## The AI screen and its limits

Every submission PR is auto-reviewed by a Claude agent that looks for exfiltration
attempts, network use, harness tampering, eval-set contact, and rule violations, and
posts its findings publicly. Two honest caveats, by design:

1. The agent is **advisory**. It gates nothing. A human maintainer reads its output,
   reads the code, and is the only one who can trigger execution (`/verify`).
2. Submissions could attempt prompt injection against the agent. It's instructed to
   treat diff content as data and to flag injection attempts — but the security of the
   system never depends on the agent being right, because execution is sandboxed and
   secretless regardless.

## What a malicious submission *can* still do

Burn bounded sandbox GPU-time until the timeout kills it, and waste a maintainer's
review time. Both are acceptable losses; repeat offenders are banned.

## Reporting

Use GitHub's private vulnerability reporting on this repo. Please don't test exploits
against live verification runs of other people's submissions.
