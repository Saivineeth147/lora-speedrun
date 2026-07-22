"""Official verification on Modal — untrusted submissions run in network-blocked GPU sandboxes.

Multi-track: each submission's config.yaml declares its track (t1 = GSM8K/Qwen2.5-1.5B,
t2 = SQuAD/SmolLM2-1.7B). The track selects the spec file, dataset dirs, pins, and
evaluator; hardware, caps, and the verification protocol are identical across tracks.

Security model (full write-up in SECURITY.md):
- The submission's train.py executes inside a Modal Sandbox: isolated VM, no secrets,
  block_network=True.
- Evaluation runs in a SEPARATE fresh sandbox that first re-verifies content hashes of the
  cached base model and dataset against committed pins (harness/integrity_check.py).
- The Modal token lives only where this orchestrator runs, never inside any sandbox.

Usage (after `pip install modal pyyaml` and `modal setup`):
  python harness/modal_verify.py --prefetch [--track t2]
  python harness/modal_verify.py --submission submissions/000-baseline --runs 3
      [--seeds 1,2,3] [--save-report records/verifications/000-baseline.md]
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
from run_submission import official_train_time, render_report  # noqa: E402  (shared report format + timing rule)

GPU = "L40S"
APP_NAME = "lora-speedrun"
VOLUME_NAME = "lora-speedrun-cache"
CACHE = "/cache"
HF_HOME = f"{CACHE}/hf"
DATA = f"{CACHE}/data"
OUT = f"{CACHE}/out"

BASE_DEPS = ["torch>=2.6.0", "transformers>=4.51.0", "peft>=0.15.0", "datasets>=3.2.0",
             "accelerate>=1.4.0", "safetensors>=0.4.5", "huggingface_hub>=0.28.0",
             "pyyaml>=6.0", "bitsandbytes>=0.45.0"]


def spec_file_for(track: str) -> str:
    return "spec.yaml" if track == "t1" else f"spec-{track}.yaml"


def load_spec(track: str) -> dict:
    path = REPO_ROOT / spec_file_for(track)
    if not path.exists():
        raise SystemExit(f"unknown track '{track}' — no {path.name} in repo root")
    return yaml.safe_load(path.read_text())


def offline_env(track: str) -> str:
    # expandable_segments prevents allocator-fragmentation OOM warnings; identical for all.
    return (f"export HF_HOME={HF_HOME} LORA_SPEEDRUN_DATA_DIR={DATA} "
            f"LORA_SPEEDRUN_SPEC=/repo/{spec_file_for(track)} "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1")


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


def do_prefetch(modal, app, image, volume, track):
    """Trusted code only; network ON so it can download the pinned model + dataset."""
    spec = load_spec(track)
    pins_name = "pins.json" if track == "t1" else f"pins-{track}.json"
    sb = make_sandbox(modal, app, image, volume, network=True, timeout=3600)
    try:
        rc, lines = run_script(sb, f"""
set -e
export HF_HOME={HF_HOME} LORA_SPEEDRUN_DATA_DIR={DATA} \
       LORA_SPEEDRUN_SPEC=/repo/{spec_file_for(track)} \
       LORA_SPEEDRUN_PINS_OUT=/tmp/pins.json
python /repo/harness/prefetch.py
echo PINS_START; cat /tmp/pins.json 2>/dev/null || cat /repo/harness/{pins_name}; echo PINS_END
echo FREEZE_START; pip freeze; echo FREEZE_END
""", f"prefetch:{track}", exec_timeout=3500)
        if rc != 0:
            raise SystemExit(f"prefetch failed (exit {rc})")

        pins_text = sentinel_block(lines, "PINS_START", "PINS_END")
        pins_path = REPO_ROOT / "harness" / pins_name
        if pins_text and not pins_path.exists():
            pins_path.write_text(pins_text + "\n")
            print(f"\nWrote {pins_path} — COMMIT THIS at spec freeze.")

        freeze = sentinel_block(lines, "FREEZE_START", "FREEZE_END")
        lock_path = REPO_ROOT / "env.lock"
        if freeze and not lock_path.exists():
            lock_path.write_text(freeze + "\n")
            print(f"Wrote {lock_path} — COMMIT THIS; future sandbox images build from it.")
        print(f"\nPrefetch complete for track {track} ({spec['base_model']}).")
    finally:
        sb.terminate()


def train_run(modal, app, image, volume, spec, track, submission_rel, seed, outdir):
    """UNTRUSTED code — network-blocked sandbox, no secrets. Timed around the exec."""
    prefix = spec["dataset"].get("local_prefix", "gsm8k")
    cap_s = spec["constraints"]["max_train_wallclock_minutes"] * 60
    sb = make_sandbox(modal, app, image, volume, gpu=GPU, network=False,
                      timeout=cap_s + 1800)
    try:
        t0 = time.monotonic()
        rc, _ = run_script(sb, f"""
set -e
{offline_env(track)}
mkdir -p {outdir}
cd /repo
python {submission_rel}/train.py --data-dir {DATA}/{prefix}_train --output-dir {outdir} --seed {seed}
""", f"train:{seed}", exec_timeout=cap_s + 600)
        train_seconds = time.monotonic() - t0
        return rc, round(train_seconds, 1)
    finally:
        sb.terminate()


def eval_run(modal, app, image, volume, spec, track, outdir):
    """Trusted code in a FRESH sandbox: integrity check, adapter audit, fixed eval."""
    prefix = spec["dataset"].get("local_prefix", "gsm8k")
    evaluator = spec["eval"].get("evaluator", "gsm8k")
    sb = make_sandbox(modal, app, image, volume, gpu=GPU, network=False, timeout=3600)
    try:
        rc, lines = run_script(sb, f"""
set -e
{offline_env(track)}
python /repo/harness/integrity_check.py
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/repo/harness")
from run_submission import count_adapter_params
print("ADAPTER_PARAMS:", count_adapter_params(Path("{outdir}")))
PY
python /repo/harness/evaluate_{evaluator}.py --data-dir {DATA}/{prefix}_test \
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
    ap.add_argument("--track", default=None, help="track for --prefetch (default t1)")
    ap.add_argument("--submission", type=Path)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--save-report", type=Path, default=None)
    args = ap.parse_args()

    modal = get_modal()
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = build_image(modal)

    if args.prefetch:
        do_prefetch(modal, app, image, volume, args.track or "t1")
        return
    if not args.submission:
        raise SystemExit("need --prefetch or --submission")

    submission = args.submission
    if not (REPO_ROOT / submission / "train.py").exists():
        raise SystemExit(f"{submission}/train.py not found")
    sub_cfg = yaml.safe_load((REPO_ROOT / submission / "config.yaml").read_text())
    track = args.track or sub_cfg.get("track", "t1")
    spec = load_spec(track)
    submission_rel = str(submission)

    n_runs = args.runs or spec["verification"]["n_runs"]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [pysecrets.randbelow(2**31) for _ in range(n_runs)])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cap = spec["constraints"]["max_trainable_params"]
    limit_s = spec["constraints"]["max_train_wallclock_minutes"] * 60

    print(f"Verifying {submission_rel} · track {track} ({spec['track']}) · "
          f"Modal {GPU} · {n_runs} runs · seeds {seeds}")
    runs = []
    for i, seed in enumerate(seeds):
        outdir = f"{OUT}/{track}/{stamp}/run{i}"
        rc, train_seconds = train_run(modal, app, image, volume, spec, track,
                                      submission_rel, seed, outdir)
        if rc != 0:
            runs.append({"seed": seed, "train_seconds": train_seconds,
                         "error": f"train.py exited {rc}", "pass": False})
            continue
        ev = eval_run(modal, app, image, volume, spec, track, outdir)
        r = {"seed": seed, "train_seconds": train_seconds,
             "accuracy": ev.get("accuracy"), "adapter_params": ev.get("adapter_params"),
             "target": spec["target_accuracy"],
             "cleared_target": (ev.get("accuracy") or 0) >= spec["target_accuracy"],
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
    official_seconds, infra_dropped = official_train_time(runs) if all_pass else (None, [])
    verdict = {
        "n_passed": n_passed,
        "all_runs_passed": all_pass,
        "mean_train_seconds": official_seconds,
        "infra_dropped_runs": infra_dropped,
        "verdict": ("RECORD-ELIGIBLE (all runs passed)" if all_pass
                    else "NOT RECORD-ELIGIBLE (a run failed — see table)"),
    }
    fp = {"gpu": f"Modal Sandbox · NVIDIA {GPU} (network-blocked)",
          "host": "modal.com", "python": sys.version.split()[0],
          "harness_commit": "(orchestrator checkout)"}

    report = {"submission": submission.name, "track": track,
              "spec_version": spec["version"], "date": date.today().isoformat(),
              "hardware": fp, "seeds": seeds, "runs": runs, "verdict": verdict}
    out_root = REPO_ROOT / "runs" / f"modal-{submission.name}-{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    md = render_report(submission.name, runs, fp, verdict, spec=spec)
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
