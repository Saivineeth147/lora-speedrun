"""Prefetch the frozen base model and dataset so timed runs execute fully offline.

Run once per machine (scripts/setup_gpu.sh does this). Writes harness/pins.json with
the resolved model/dataset revisions and a content hash of the train split; at spec
freeze that file is committed and verification enforces it thereafter.

Usage: python harness/prefetch.py
"""

import hashlib
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
PINS_PATH = REPO_ROOT / "harness" / "pins.json"
DATA_DIR = REPO_ROOT / "data"


def main():
    spec = yaml.safe_load((REPO_ROOT / "spec.yaml").read_text())
    base_model = spec["base_model"]
    ds_repo = spec["dataset"]["repo"]
    ds_config = spec["dataset"]["config"]

    existing_pins = json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else {}
    api = HfApi()

    # If pins are committed (post-freeze), download exactly those revisions.
    model_rev = existing_pins.get("base_model_sha")
    print(f"Downloading {base_model} (revision: {model_rev or 'latest'}) ...")
    snapshot_download(base_model, revision=model_rev)
    model_sha = model_rev or api.model_info(base_model).sha

    ds_rev = existing_pins.get("dataset_sha")
    print(f"Downloading {ds_repo}/{ds_config} (revision: {ds_rev or 'latest'}) ...")
    ds = load_dataset(ds_repo, ds_config, revision=ds_rev)
    dataset_sha = ds_rev or api.dataset_info(ds_repo).sha

    train, test = ds[spec["dataset"]["train_split"]], ds[spec["dataset"]["eval_split"]]
    DATA_DIR.mkdir(exist_ok=True)
    # Saved separately: timed training runs are handed ONLY the train directory.
    train.save_to_disk(str(DATA_DIR / "gsm8k_train"))
    test.save_to_disk(str(DATA_DIR / "gsm8k_test"))

    h = hashlib.sha256()
    for ex in train:
        h.update(ex["question"].encode())
        h.update(ex["answer"].encode())
    pins = {
        "base_model": base_model,
        "base_model_sha": model_sha,
        "dataset": ds_repo,
        "dataset_sha": dataset_sha,
        "train_examples": len(train),
        "test_examples": len(test),
        "train_content_sha256": h.hexdigest(),
    }

    if existing_pins:
        for key in ("base_model_sha", "train_content_sha256", "train_examples"):
            if pins[key] != existing_pins.get(key):
                raise SystemExit(
                    f"PIN MISMATCH on {key}: got {pins[key]}, "
                    f"expected {existing_pins.get(key)}. Refusing to continue."
                )
        print("Pins verified against committed harness/pins.json ✓")
    else:
        PINS_PATH.write_text(json.dumps(pins, indent=2) + "\n")
        print(f"Wrote {PINS_PATH} — commit this file at spec freeze.")

    print(json.dumps(pins, indent=2))
    print(f"\nPrefetch complete. Train data: {DATA_DIR/'gsm8k_train'} ({len(train)} ex), "
          f"test data: {DATA_DIR/'gsm8k_test'} ({len(test)} ex)")


if __name__ == "__main__":
    main()
