"""Timed verification runner — the only clock that counts.

For each run: launches the submission's train.py with a fresh seed (downloads pre-cached,
network off), times the whole process, audits the saved adapter's parameter count, then
evaluates base+adapter with the fixed evaluator. Emits report.json plus a markdown
verification report ready to paste into the PR.

Usage:
  python harness/run_submission.py submissions/000-baseline --runs 3
  python harness/run_submission.py submissions/007-handle --runs 3 \
      --save-report records/verifications/007-handle.md
"""

import argparse
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec_path = Path(os.environ.get("LORA_SPEEDRUN_SPEC", REPO_ROOT / "spec.yaml"))
SPEC = yaml.safe_load((_spec_path if _spec_path.is_absolute() else REPO_ROOT / _spec_path).read_text())
DATA_PREFIX = SPEC["dataset"].get("local_prefix", "gsm8k")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unavailable"


def hardware_fingerprint():
    return {
        "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                   "--format=csv,noheader"]),
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "harness_commit": sh(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]),
    }


def count_adapter_params(adapter_dir: Path) -> int:
    """Every tensor saved in the adapter counts against the cap — including
    modules_to_save full-layer copies. This is what stops 'adapters' that are
    actually full fine-tunes."""
    from safetensors import safe_open

    files = sorted(adapter_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors adapter found in {adapter_dir}")
    total = 0
    for f in files:
        with safe_open(str(f), framework="pt") as st:
            for key in st.keys():
                n = 1
                for dim in st.get_slice(key).get_shape():
                    n *= dim
                total += n
    return total


def one_run(submission_dir: Path, seed: int, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1",
               # avoid allocator fragmentation -> transient OOM warnings; no effect on results
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, str(submission_dir / "train.py"),
           "--data-dir", str(REPO_ROOT / "data" / f"{DATA_PREFIX}_train"),
           "--output-dir", str(run_dir),
           "--seed", str(seed)]
    print(f"\n=== TRAIN seed={seed} -> {run_dir}\n$ {' '.join(cmd)}", flush=True)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env)
    train_seconds = time.monotonic() - t0
    if proc.returncode != 0:
        return {"seed": seed, "train_seconds": round(train_seconds, 1),
                "error": f"train.py exited {proc.returncode}", "pass": False}

    limit_s = SPEC["constraints"]["max_train_wallclock_minutes"] * 60
    over_limit = train_seconds > limit_s

    adapter_params = count_adapter_params(run_dir)
    cap = SPEC["constraints"]["max_trainable_params"]

    eval_out = run_dir / "eval.json"
    evaluator = SPEC["eval"].get("evaluator", "gsm8k")
    print(f"=== EVAL {run_dir}", flush=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "harness" / f"evaluate_{evaluator}.py"),
                    "--data-dir", str(REPO_ROOT / "data" / f"{DATA_PREFIX}_test"),
                    "--adapter-dir", str(run_dir),
                    "--out", str(eval_out)], env=env, check=True)
    accuracy = json.loads(eval_out.read_text())["accuracy"]

    result = {
        "seed": seed,
        "train_seconds": round(train_seconds, 1),
        "accuracy": round(accuracy, 4),
        "adapter_params": adapter_params,
        "target": SPEC["target_accuracy"],
        "cleared_target": accuracy >= SPEC["target_accuracy"],
        "within_param_cap": adapter_params <= cap,
        "within_time_limit": not over_limit,
    }
    result["pass"] = (result["cleared_target"] and result["within_param_cap"]
                      and result["within_time_limit"])
    return result


def render_report(submission: str, runs: list, fp: dict, verdict: dict, spec: dict = None) -> str:
    SPEC = spec or globals()["SPEC"]
    rows = "\n".join(
        f"| {i+1} | {r['seed']} | {r['train_seconds']:.1f}s | "
        f"{r.get('accuracy', '—')} | {'✅' if r['pass'] else '❌'} |"
        for i, r in enumerate(runs))
    params = next((r["adapter_params"] for r in runs if "adapter_params" in r), "—")
    return f"""# Verification Report — {submission}

- **Date:** {date.today().isoformat()}
- **Verifier:** (fill in GitHub handle; mark `self-verified` if verifying own submission)
- **PR:** #
- **Hardware:** {fp['gpu']}
- **Harness commit:** {fp['harness_commit']} · Python {fp['python']}
- **Target:** ≥ {SPEC['target_accuracy']:.1%} GSM8K (spec v{SPEC['version']})
- **Seeds:** fresh, generated at verification time (below)

## Timed runs

| run | seed | train wall-clock | GSM8K acc | pass |
|-----|------|------------------|-----------|------|
{rows}

**Mean train time (all-pass): {verdict['mean_train_seconds'] or '—'}s** ·
runs passed: {verdict['n_passed']}/{len(runs)}

## Adapter audit

- Trainable params in saved adapter: **{params:,}** (cap {SPEC['constraints']['max_trainable_params']:,}) —
  {'within cap ✅' if all(r.get('within_param_cap', False) for r in runs) else 'OVER CAP ❌'}
- Loads in pinned peft against frozen base: (verifier confirms)

## Code review notes

- Training data sources verified train-split-only: (verifier fills in)
- Technique summary: (verifier fills in)
- Concerns: (verifier fills in, or "none")

## Verdict

**{verdict['verdict']}** — (verifier adds reasoning: what the technique does, why it's
legitimate, how it compares to the current record.)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_dir", type=Path)
    ap.add_argument("--runs", type=int, default=SPEC["verification"]["n_runs"])
    ap.add_argument("--seeds", default=None,
                    help="comma-separated; default = fresh random seeds, recorded in report")
    ap.add_argument("--save-report", type=Path, default=None,
                    help="also write the markdown verification report here")
    args = ap.parse_args()

    submission_dir = args.submission_dir.resolve()
    if not (submission_dir / "train.py").exists():
        raise SystemExit(f"{submission_dir} has no train.py")
    if not (REPO_ROOT / "data" / f"{DATA_PREFIX}_train").exists():
        raise SystemExit("data/ not prefetched — run: python harness/prefetch.py")

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [secrets.randbelow(2**31) for _ in range(args.runs)])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = REPO_ROOT / "runs" / f"{submission_dir.name}-{stamp}"
    fp = hardware_fingerprint()
    print(f"Verifying {submission_dir.name} · {args.runs} runs · seeds {seeds}")
    print(f"Hardware: {fp['gpu']}")

    runs = [one_run(submission_dir, seed, out_root / f"run{i}")
            for i, seed in enumerate(seeds)]

    n_passed = sum(r["pass"] for r in runs)
    all_pass = n_passed == len(runs)
    mean_t = round(sum(r["train_seconds"] for r in runs) / len(runs), 1) if all_pass else None
    verdict = {
        "n_passed": n_passed,
        "all_runs_passed": all_pass,
        "mean_train_seconds": mean_t,
        "verdict": "RECORD-ELIGIBLE (all runs passed)" if all_pass
                   else "NOT RECORD-ELIGIBLE (a run failed — see table)",
    }

    report = {"submission": submission_dir.name, "spec_version": SPEC["version"],
              "hardware": fp, "seeds": seeds, "runs": runs, "verdict": verdict}
    (out_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    md = render_report(submission_dir.name, runs, fp, verdict)
    (out_root / "report.md").write_text(md)
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(md)

    print("\n" + "=" * 72)
    print(md)
    print(f"Artifacts: {out_root}")


if __name__ == "__main__":
    main()
