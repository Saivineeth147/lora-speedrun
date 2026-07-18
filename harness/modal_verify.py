"""Official verification on Modal — untrusted submissions run in network-blocked GPU sandboxes.

Security model (full write-up in SECURITY.md):
- The submission's train.py executes inside a Modal Sandbox: isolated VM, no secrets,
  block_network=True. It cannot reach the network or anything of ours.
- Evaluation runs in a SEPARATE fresh sandbox that first re-verifies content hashes of the
  cached base model and dataset against committed pins (harness/integrity_check.py) —
  a malicious training run that poisons the shared cache volume gets caught, not scored.
- The Modal token lives only where this orchestrator runs (your machine / GitHub Actions
  secrets). No sandbox ever sees it.

Usage (after `pip install modal pyyaml` and `modal setup`):
  python harness/modal_verify.py --prefetch                 # one-time: populate cache volume,
                                                            # emit harness/pins.json + env.lock
  python harness/modal_verify.py --submission submissions/000-baseline --runs 3
      [--seeds 1,2,3] [--save-report records/verifications/000-baseline.md]

Environment pinning: once env.lock exists (written by --prefetch), the sandbox image is
built from it, so every verification uses an identical dependency set.
"""

import argparse
import json
import secrets as pysecrets
import sys
import time
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "harness"))
from run_submission import render_report  # noqa: E402  (shares SPEC + report format)

SPEC = yaml.safe_load((REPO_ROOT / "spec.yaml").read_text())

GPU = "L40S"
APP_NAME = "lora-speedrun"
VOLUME_NAME = "lora-speedrun-cache"
CACHE = "/cache"
HF_HOME = f"{CACHE}/hf"
DATA = f"{CACHE}/data"
OUT = f"{CACHE}/out"

OFFLINE_ENV = (f"export HF_HOME={HF_HOME} LORA_SPEEDRUN_DATA_DIR={DATA} "
               f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1")

BASE_DEPS = ["torch>=2.6.0", "transformers>=4.51.0", "peft>=0.15.0", "datasets>=3.2.0",
             "accelerate>=1.4.0", "safetensors>=0.4.5", "huggingface_hub>=0.28.0",
             "pyyaml>=6.0", "bitsandbytes>=0.45.0"]


def get_modal():
    try:
        import modal
        return modal
    except ImportError:
        raise SystemExit("modal not installed — run: pip install modal && modal setup")


def build_image(modal):
    lock = REPO_ROOT / "env.lock"
    img = modal.Image.debian_slim(python_version="3.12")
    img = (img.pip_install_from_requirements(str(lock)) if lock.exists()
           else img.pip_install(*BASE_DEPS))
    return img.add_local_dir(
        str(REPO_ROOT), remote_path="/repo",
        ignore=["**/.git/**", "**/__pycache__/**", "runs/**", "data/**", "**/.DS_Store"])


def make_sandbox(modal, app, image, volume, *, gpu=None, network=False, timeout=3600):
    return modal.Sandbox.create(
        app=app, image=image, gpu=gpu,
        volumes={CACHE: volume},
        block_network=not network,
        timeout=timeout)


def run_script(sb, script, tag, exec_timeout=None):
    """Exec a bash script in the sandbox, streaming output. Returns (returncode, lines)."""
    p = sb.exec("bash", "-c", f"exec 2>&1\n{script}",
                **({"timeout": exec_timeout} if exec_timeout else {}))
    lines = []
    for line in p.stdout:
        lines.append(line)
        print(f"[{tag}] {line}", end="", flush=True)
    p.wait()
    return p.returncode, lines


def sentinel_block(lines, start, end):
    text = "".join(lines)
    if start in text and end in text:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    return None


def do_prefetch(modal, app, image, volume):
    """Trusted code only; network ON so it can download the pinned model + dataset."""
    sb = make_sandbox(modal, app, image, volume, network=True, timeout=3600)
    try:
        rc, lines = run_script(sb, f"""
set -e
export HF_HOME={HF_HOME} LORA_SPEEDRUN_DATA_DIR={DATA} LORA_SPEEDRUN_PINS_OUT=/tmp/pins.json
python /repo/harness/prefetch.py
echo PINS_START; cat /tmp/pins.json 2>/dev/null || cat /repo/harness/pins.json; echo PINS_END
echo FREEZE_START; pip freeze; echo FREEZE_END
""", "prefetch", exec_timeout=3500)
        if rc != 0:
            raise SystemExit(f"prefetch failed (exit {rc})")

        pins_text = sentinel_block(lines, "PINS_START", "PINS_END")
        pins_path = REPO_ROOT / "harness" / "pins.json"
        if pins_text and not pins_path.exists():
            pins_path.write_text(pins_text + "\n")
            print(f"\nWrote {pins_path} — COMMIT THIS at spec freeze.")

        freeze = sentinel_block(lines, "FREEZE_START", "FREEZE_END")
        lock_path = REPO_ROOT / "env.lock"
        if freeze and not lock_path.exists():
            lock_path.write_text(freeze + "\n")
            print(f"Wrote {lock_path} — COMMIT THIS; future sandbox images build from it.")
        print("\nPrefetch complete. Cache volume is populated.")
    finally:
        sb.terminate()


def train_run(modal, app, image, volume, submission_rel, seed, outdir):
    """UNTRUSTED code — network-blocked sandbox, no secrets. Timed around the exec."""
    cap_s = SPEC["constraints"]["max_train_wallclock_minutes"] * 60
    sb = make_sandbox(modal, app, image, volume, gpu=GPU, network=False,
                      timeout=cap_s + 1800)
    try:
        t0 = time.monotonic()
        rc, _ = run_script(sb, f"""
set -e
{OFFLINE_ENV}
mkdir -p {outdir}
cd /repo
python {submission_rel}/train.py --data-dir {DATA}/gsm8k_train --output-dir {outdir} --seed {seed}
""", f"train:{seed}", exec_timeout=cap_s + 600)
        train_seconds = time.monotonic() - t0
        return rc, round(train_seconds, 1)
    finally:
        sb.terminate()


def eval_run(modal, app, image, volume, outdir):
    """Trusted code in a FRESH sandbox: integrity check, adapter audit, fixed eval."""
    sb = make_sandbox(modal, app, image, volume, gpu=GPU, network=False, timeout=3600)
    try:
        rc, lines = run_script(sb, f"""
set -e
{OFFLINE_ENV}
python /repo/harness/integrity_check.py
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/repo/harness")
from run_submission import count_adapter_params
print("ADAPTER_PARAMS:", count_adapter_params(Path("{outdir}")))
PY
python /repo/harness/evaluate_gsm8k.py --data-dir {DATA}/gsm8k_test \
    --adapter-dir {outdir} --out {outdir}/eval.json
echo EVAL_JSON_START; cat {outdir}/eval.json; echo; echo EVAL_JSON_END
""", "eval", exec_timeout=3500)
        if rc != 0:
            return {"error": f"eval/integrity failed (exit {rc}) — see log above"}

        adapter_params = None
        for line in lines:
            if line.startswith("ADAPTER_PARAMS:"):
                adapter_params = int(line.split(":", 1)[1].strip())
        eval_json = sentinel_block(lines, "EVAL_JSON_START", "EVAL_JSON_END")
        accuracy = json.loads(eval_json)["accuracy"] if eval_json else None
        return {"adapter_params": adapter_params, "accuracy": accuracy}
    finally:
        sb.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--submission", type=Path)
    ap.add_argument("--runs", type=int, default=SPEC["verification"]["n_runs"])
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--save-report", type=Path, default=None)
    args = ap.parse_args()

    modal = get_modal()
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = build_image(modal)

    if args.prefetch:
        do_prefetch(modal, app, image, volume)
        return
    if not args.submission:
        raise SystemExit("need --prefetch or --submission")

    submission = args.submission
    if not (REPO_ROOT / submission / "train.py").exists():
        raise SystemExit(f"{submission}/train.py not found")
    submission_rel = str(submission)

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [pysecrets.randbelow(2**31) for _ in range(args.runs)])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cap = SPEC["constraints"]["max_trainable_params"]
    limit_s = SPEC["constraints"]["max_train_wallclock_minutes"] * 60

    print(f"Verifying {submission_rel} on Modal {GPU} · {args.runs} runs · seeds {seeds}")
    runs = []
    for i, seed in enumerate(seeds):
        outdir = f"{OUT}/{stamp}/run{i}"
        rc, train_seconds = train_run(modal, app, image, volume, submission_rel, seed, outdir)
        if rc != 0:
            runs.append({"seed": seed, "train_seconds": train_seconds,
                         "error": f"train.py exited {rc}", "pass": False})
            continue
        ev = eval_run(modal, app, image, volume, outdir)
        r = {"seed": seed, "train_seconds": train_seconds,
             "accuracy": ev.get("accuracy"), "adapter_params": ev.get("adapter_params"),
             "target": SPEC["target_accuracy"],
             "cleared_target": (ev.get("accuracy") or 0) >= SPEC["target_accuracy"],
             "within_param_cap": (ev.get("adapter_params") or cap + 1) <= cap,
             "within_time_limit": train_seconds <= limit_s}
        if "error" in ev:
            r["error"] = ev["error"]
        r["pass"] = (r["cleared_target"] and r["within_param_cap"]
                     and r["within_time_limit"] and "error" not in r)
        runs.append(r)
        print(f"--- run {i+1}/{len(seeds)}: {r['train_seconds']}s, "
              f"acc={r.get('accuracy')}, pass={r['pass']}")

    n_passed = sum(r["pass"] for r in runs)
    all_pass = n_passed == len(runs)
    verdict = {
        "n_passed": n_passed,
        "all_runs_passed": all_pass,
        "mean_train_seconds": (round(sum(r["train_seconds"] for r in runs) / len(runs), 1)
                               if all_pass else None),
        "verdict": ("RECORD-ELIGIBLE (all runs passed)" if all_pass
                    else "NOT RECORD-ELIGIBLE (a run failed — see table)"),
    }
    fp = {"gpu": f"Modal Sandbox · NVIDIA {GPU} (network-blocked)",
          "host": "modal.com", "python": sys.version.split()[0],
          "harness_commit": "(orchestrator checkout)"}

    report = {"submission": submission.name, "spec_version": SPEC["version"],
              "date": date.today().isoformat(), "hardware": fp,
              "seeds": seeds, "runs": runs, "verdict": verdict}
    out_root = REPO_ROOT / "runs" / f"modal-{submission.name}-{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    md = render_report(submission.name, runs, fp, verdict)
    (out_root / "report.md").write_text(md)
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(md)

    print("\n" + "=" * 72)
    print(md)
    print(f"Artifacts: {out_root}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
