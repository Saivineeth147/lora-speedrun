#!/usr/bin/env bash
# One-shot setup on your OWN GPU box (any 24GB+ CUDA card) for LOCAL ITERATION.
# Official leaderboard timing runs on Modal (harness/modal_verify.py) — local times
# are for experimenting only. Installs deps, prefetches model + data.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  echo "No NVIDIA GPU visible — this must run on the spec hardware."; exit 1; }

echo "== Installing requirements =="
if [ -f env.lock ]; then
  echo "env.lock found — installing exact frozen environment"
  pip install -r env.lock
else
  pip install -r requirements.txt
  pip freeze > env.lock
  echo "Wrote env.lock — commit it at spec freeze so verification envs are identical."
fi

echo "== Prefetching model + dataset (not part of timed runs) =="
python harness/prefetch.py

echo "
Setup complete. Next:
  # quick smoke test of the evaluator (32 problems, base model — expect a low score):
  python harness/evaluate_gsm8k.py --data-dir data/gsm8k_test --limit 32

  # one full timed baseline attempt:
  python harness/run_submission.py submissions/000-baseline --runs 1

  # record-style verification (3 fresh seeds):
  python harness/run_submission.py submissions/000-baseline --runs 3 \\
      --save-report records/verifications/000-baseline.md
"
