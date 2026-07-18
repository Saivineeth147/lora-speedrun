"""Verify the cached base model + dataset against committed pins (harness/pins.json).

Runs inside the fresh eval sandbox before every evaluation. This is what makes the
shared cache volume safe even though untrusted training code can write to it: if a
malicious run tampered with the base model weights or the eval data, hashes won't
match and the verification aborts instead of scoring a poisoned setup.

Exit codes: 0 = ok/skipped (pre-freeze), 2 = INTEGRITY FAILURE.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((REPO_ROOT / "spec.yaml").read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def dataset_content_sha(ds_dir: Path) -> str:
    from datasets import load_from_disk
    ds = load_from_disk(str(ds_dir))
    h = hashlib.sha256()
    for ex in ds:
        h.update(ex["question"].encode())
        h.update(ex["answer"].encode())
    return h.hexdigest()


def main():
    pins_path = REPO_ROOT / "harness" / "pins.json"
    if not pins_path.exists():
        print("INTEGRITY: SKIPPED (harness/pins.json not committed yet — pre-freeze mode)")
        return
    pins = json.loads(pins_path.read_text())
    failures = []

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_repo_dir = hf_home / "hub" / ("models--" + SPEC["base_model"].replace("/", "--"))
    snapshot = model_repo_dir / "snapshots" / pins["base_model_sha"]
    if not snapshot.exists():
        snapshots = sorted((model_repo_dir / "snapshots").glob("*")) if model_repo_dir.exists() else []
        if snapshots:
            snapshot = snapshots[-1]
        else:
            failures.append("base model snapshot missing from cache")

    for fname, want in pins.get("model_file_sha256", {}).items():
        f = snapshot / fname
        if not f.exists():
            failures.append(f"model file missing: {fname}")
        elif sha256_file(f) != want:
            failures.append(f"model file hash mismatch: {fname}")

    data_root = Path(os.environ.get("LORA_SPEEDRUN_DATA_DIR", REPO_ROOT / "data"))
    for split_dir, key in (("gsm8k_train", "train_content_sha256"),
                           ("gsm8k_test", "test_content_sha256")):
        if key not in pins:
            continue
        d = data_root / split_dir
        if not d.exists():
            failures.append(f"dataset missing: {split_dir}")
        elif dataset_content_sha(d) != pins[key]:
            failures.append(f"dataset content mismatch: {split_dir}")

    if failures:
        print("INTEGRITY: FAILED — " + "; ".join(failures))
        print("The cache volume may have been tampered with. Wipe it and re-run "
              "`python harness/modal_verify.py --prefetch`, then re-verify.")
        sys.exit(2)
    print("INTEGRITY: OK (base model + train/test data match committed pins)")


if __name__ == "__main__":
    main()
