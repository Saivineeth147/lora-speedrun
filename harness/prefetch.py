"""Prefetch the frozen base model and dataset so timed runs execute fully offline.

Run once per machine — locally via scripts/setup_gpu.sh, or inside the Modal cache
volume via `python harness/modal_verify.py --prefetch`. Records resolved revisions plus
content hashes (model files, train/test data) into pins.json; at spec freeze that file
is committed as harness/pins.json and integrity_check.py enforces it before every eval.

Env overrides (used by the Modal sandbox, where /repo is read-only):
  LORA_SPEEDRUN_DATA_DIR  where to save the train/test splits (default: <repo>/data)
  LORA_SPEEDRUN_PINS_OUT  where to write pins.json when none is committed yet
                          (default: <repo>/harness/pins.json)
"""

import hashlib
import json
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec_path = Path(os.environ.get("LORA_SPEEDRUN_SPEC", REPO_ROOT / "spec.yaml"))
SPEC = yaml.safe_load((_spec_path if _spec_path.is_absolute() else REPO_ROOT / _spec_path).read_text())
TRACK = SPEC.get("track_id", "t1")
PINS_NAME = "pins.json" if TRACK == "t1" else f"pins-{TRACK}.json"
HASH_SCHEME = "qa-v1" if TRACK == "t1" else "generic"
DATA_PREFIX = SPEC["dataset"].get("local_prefix", "gsm8k")

COMMITTED_PINS = REPO_ROOT / "harness" / PINS_NAME
DATA_DIR = Path(os.environ.get("LORA_SPEEDRUN_DATA_DIR", REPO_ROOT / "data"))
PINS_OUT = Path(os.environ.get("LORA_SPEEDRUN_PINS_OUT", COMMITTED_PINS))

PINNED_MODEL_FILES = ("*.safetensors", "tokenizer.json", "config.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def dataset_content_sha(ds) -> str:
    h = hashlib.sha256()
    for ex in ds:
        if HASH_SCHEME == "qa-v1":
            h.update(ex["question"].encode())
            h.update(ex["answer"].encode())
        else:  # generic: stable JSON of the whole example
            h.update(json.dumps(ex, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def main():
    spec = SPEC
    base_model = spec["base_model"]
    ds_repo, ds_config = spec["dataset"]["repo"], spec["dataset"]["config"]

    existing = json.loads(COMMITTED_PINS.read_text()) if COMMITTED_PINS.exists() else {}
    api = HfApi()

    model_rev = existing.get("base_model_sha")
    print(f"Downloading {base_model} (revision: {model_rev or 'latest'}) ...")
    snapshot_path = Path(snapshot_download(base_model, revision=model_rev))
    model_sha = model_rev or api.model_info(base_model).sha

    model_hashes = {}
    for pattern in PINNED_MODEL_FILES:
        for f in sorted(snapshot_path.glob(pattern)):
            model_hashes[f.name] = sha256_file(f)

    ds_rev = existing.get("dataset_sha")
    print(f"Downloading {ds_repo}/{ds_config} (revision: {ds_rev or 'latest'}) ...")
    ds = load_dataset(ds_repo, ds_config, revision=ds_rev)
    dataset_sha = ds_rev or api.dataset_info(ds_repo).sha

    train = ds[spec["dataset"]["train_split"]]
    test = ds[spec["dataset"]["eval_split"]]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Saved separately: timed training runs are handed ONLY the train directory.
    train.save_to_disk(str(DATA_DIR / f"{DATA_PREFIX}_train"))
    test.save_to_disk(str(DATA_DIR / f"{DATA_PREFIX}_test"))

    pins = {
        "track_id": TRACK,
        "hash_scheme": HASH_SCHEME,
        "base_model": base_model,
        "base_model_sha": model_sha,
        "model_file_sha256": model_hashes,
        "dataset": ds_repo,
        "dataset_sha": dataset_sha,
        "train_examples": len(train),
        "test_examples": len(test),
        "train_content_sha256": dataset_content_sha(train),
        "test_content_sha256": dataset_content_sha(test),
    }

    if existing:
        mismatches = [k for k in ("base_model_sha", "train_content_sha256",
                                  "test_content_sha256", "train_examples")
                      if pins[k] != existing.get(k)]
        for fname, want in existing.get("model_file_sha256", {}).items():
            if pins["model_file_sha256"].get(fname) != want:
                mismatches.append(f"model_file_sha256:{fname}")
        if mismatches:
            raise SystemExit(f"PIN MISMATCH on {mismatches} — refusing to continue. "
                             "The frozen artifacts changed upstream or the cache is bad.")
        print("Pins verified against committed harness/pins.json ✓")
    else:
        PINS_OUT.parent.mkdir(parents=True, exist_ok=True)
        PINS_OUT.write_text(json.dumps(pins, indent=2) + "\n")
        print(f"Wrote {PINS_OUT} — commit as harness/{PINS_NAME} at spec freeze.")

    print(f"\nPrefetch complete. Train: {DATA_DIR/(DATA_PREFIX+'_train')} ({len(train)} ex) · "
          f"test: {DATA_DIR/(DATA_PREFIX+'_test')} ({len(test)} ex)")


if __name__ == "__main__":
    main()
