"""Render the Hugging Face Space page from records/records.json.

The Space used to fetch records.json cross-origin on every page load, which meant
visitors saw an empty page until that request came back — and nothing at all if it
was slow, rate-limited, or blocked. This bakes the current records into the page so
it paints immediately, and keeps the live fetch as a refresh on top.

Usage:
  python scripts/render_space.py           # write space/index.html
  python scripts/render_space.py --check   # CI: fail if space/index.html is stale
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "space" / "index.template.html"
OUTPUT = REPO_ROOT / "space" / "index.html"
RECORDS = REPO_ROOT / "records" / "records.json"

# The stamp changes every run, so --check has to compare everything except the stamp.
STAMP_TOKEN = "__BUILD_STAMP__"
RECORDS_TOKEN = "__RECORDS_JSON__"


def render(stamp: str) -> str:
    data = json.loads(RECORDS.read_text())
    # Drop the maintainer comment; it is noise in a page payload.
    data.pop("_comment", None)
    # "</script>" inside the JSON would close the tag early and break the page.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace(RECORDS_TOKEN, blob).replace(STAMP_TOKEN, stamp)


def strip_stamp(html: str) -> str:
    """Everything except the build timestamp, for drift comparison."""
    head, sep, tail = html.partition("records baked in at ")
    if not sep:
        return html
    _, close, rest = tail.partition("</span>")
    return head + close + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    new = render(stamp)

    if args.check:
        if not OUTPUT.exists():
            print("DRIFT: space/index.html is missing.\nRun: python scripts/render_space.py")
            sys.exit(1)
        if strip_stamp(OUTPUT.read_text()) != strip_stamp(new):
            print("DRIFT: space/index.html is out of date with records/records.json.\n"
                  "Run: python scripts/render_space.py")
            sys.exit(1)
        print("space page in sync ✓")
        return

    OUTPUT.write_text(new)
    n = len(json.loads(RECORDS.read_text())["records"])
    print(f"space/index.html regenerated ({n} records baked in)")


if __name__ == "__main__":
    main()
