"""Static submission validator — runs in CI on every PR. CPU-only, no ML deps.

Checks structure and rules that don't need a GPU: required files, config schema,
base model matches spec, train.py parses, no weight files smuggled into the repo.

Usage:
  python harness/validate_submission.py --all
  python harness/validate_submission.py submissions/007-handle
"""

import argparse
import ast
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS = {}
for f in REPO_ROOT.glob("spec*.yaml"):
    s = yaml.safe_load(f.read_text())
    SPECS[s.get("track_id", "t1")] = s

REQUIRED_FILES = ["train.py", "config.yaml", "NOTES.md"]
REQUIRED_CONFIG_KEYS = ["author", "github", "description", "base_model", "technique"]
BANNED_EXTENSIONS = {".safetensors", ".bin", ".pt", ".gguf", ".ckpt"}
MAX_FILE_BYTES = 5 * 1024 * 1024


def validate(sub_dir: Path) -> list:
    errors = []
    for name in REQUIRED_FILES:
        if not (sub_dir / name).exists():
            errors.append(f"missing required file: {name}")
    if errors:
        return errors

    try:
        cfg = yaml.safe_load((sub_dir / "config.yaml").read_text())
        if not isinstance(cfg, dict):
            raise ValueError("config.yaml is not a mapping")
    except Exception as e:
        return errors + [f"config.yaml unreadable: {e}"]

    for key in REQUIRED_CONFIG_KEYS:
        if not cfg.get(key):
            errors.append(f"config.yaml missing/empty required key: {key}")
    track = cfg.get("track", "t1")
    if track not in SPECS:
        errors.append(f"unknown track '{track}' (available: {sorted(SPECS)})")
    elif cfg.get("base_model") and cfg["base_model"] != SPECS[track]["base_model"]:
        errors.append(f"base_model for track {track} must be exactly "
                      f"'{SPECS[track]['base_model']}' (got '{cfg['base_model']}')")

    try:
        ast.parse((sub_dir / "train.py").read_text())
    except SyntaxError as e:
        errors.append(f"train.py has a syntax error: line {e.lineno}: {e.msg}")

    notes = (sub_dir / "NOTES.md").read_text().strip()
    if len(notes) < 100:
        errors.append("NOTES.md is effectively empty — explain the technique (see TEMPLATE)")

    for f in sub_dir.rglob("*"):
        if f.is_file():
            if f.suffix.lower() in BANNED_EXTENSIONS:
                errors.append(f"weight file in submission (never commit weights): {f.name}")
            elif f.stat().st_size > MAX_FILE_BYTES:
                errors.append(f"file too large (>5MB): {f.name}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="validate every submission dir")
    args = ap.parse_args()

    if args.all:
        dirs = sorted(d for d in (REPO_ROOT / "submissions").iterdir()
                      if d.is_dir() and d.name != "TEMPLATE")
    else:
        dirs = args.dirs
    if not dirs:
        print("nothing to validate")
        return

    failed = False
    for d in dirs:
        errs = validate(d)
        if errs:
            failed = True
            print(f"FAIL {d.name}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {d.name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
