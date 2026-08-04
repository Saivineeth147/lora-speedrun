"""Regenerate the Track 1 hero card (docs/leaderboard.png) from records/records.json.

The markdown leaderboards are rebuilt by scripts/update_leaderboard.py; this does the
same for the PNG card shown at the top of the README, so the image never drifts from the
data. Emits an SVG and rasterizes it with rsvg-convert (already used in this repo's
tooling). Run after editing records.json:

    python scripts/generate_hero_card.py

Bars are proportional to training wall-clock (slowest verified record = full width).
Records are shown best-on-top; the fastest verified record is highlighted in green.
"""

import html
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACK = "t1"
OUT_PNG = REPO_ROOT / "docs" / "leaderboard.png"
OUT_SVG = REPO_ROOT / "docs" / "leaderboard.svg"

W, H = 1600, 900
GREEN = "#0ca30c"
WHITE = "#ffffff"
GRAY = "#c3c2b7"       # subtitle + value text
DIM = "#7c7c72"        # meta line
BAR_TRACK = "#2c2c2a"
BAR_GRAY = "#7a7a70"   # non-record bar fill
CARD_BG = "#1a1a19"
OUTER_BG = "#0d0d0d"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

LEFT_X = 112
RIGHT_X = 1488
BAR_W = RIGHT_X - LEFT_X

# Short, card-friendly technique labels. A record may override with a "card_label"
# field in records.json; otherwise this map is used, then a truncation fallback.
CARD_LABELS = {
    ("t1", 0): "baseline, no tricks",
    ("t1", 1): "packing + loss masking",
    ("t1", 2): "3k examples, integrated-LR",
    ("t1", 3): "shortest-4k + chunked CE",
    ("t1", 4): "fused 6-target LoRA, 0.75 epoch",
    ("t1", 5): "direct load + staged backward tail",
}

# Rows are laid out between the header and the footer. Keep the original 98px pitch
# while it fits, then tighten it so a growing board never runs into the footer line.
ROW_TOP = 286
ROW_PITCH_MAX = 98
FOOTER_Y = 812
# A row's bar sits 30px below its label and is 14px tall; leave a gap under the last one.
ROW_BOTTOM_PAD = 44
FOOTER_GAP = 52


def row_pitch(n_rows):
    last_label_max = FOOTER_Y - ROW_BOTTOM_PAD - FOOTER_GAP
    return min(ROW_PITCH_MAX, (last_label_max - ROW_TOP) // max(1, n_rows - 1))


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def mean_acc(rec):
    return sum(rec["accuracies"]) / len(rec["accuracies"])


def card_label(rec):
    return (
        rec.get("card_label")
        or CARD_LABELS.get((rec.get("track", "t1"), rec["id"]))
        or (rec["description"][:38].rstrip(" ,.:") + "…")
    )


def esc(s):
    return html.escape(str(s), quote=False)


def text(x, y, s, size, fill, *, weight="normal", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def bar(y, frac, fill):
    frac = max(0.0, min(1.0, frac))
    w = max(10, round(BAR_W * frac))
    return (
        f'<rect x="{LEFT_X}" y="{y}" width="{BAR_W}" height="14" rx="7" fill="{BAR_TRACK}"/>'
        f'<rect x="{LEFT_X}" y="{y}" width="{w}" height="14" rx="7" fill="{fill}"/>'
    )


def build_svg(records):
    recs = [r for r in records if r.get("track", "t1") == TRACK
            and r["status"] == "verified" and r["time_seconds_mean"] is not None]
    recs.sort(key=lambda r: r["id"])
    slowest = max(r["time_seconds_mean"] for r in recs)
    fastest = min(r["time_seconds_mean"] for r in recs)
    record_id = next(r["id"] for r in recs if r["time_seconds_mean"] == fastest)
    rows = list(reversed(recs))  # best on top

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{OUTER_BG}"/>',
        f'<rect x="40" y="25" width="1520" height="850" rx="24" fill="{CARD_BG}" '
        f'stroke="{BAR_TRACK}" stroke-width="1.5"/>',
        text(LEFT_X, 112, "LoRA Speedrun", 76, WHITE, weight="bold"),
        text(LEFT_X, 160,
             "Fine-tune Qwen2.5-1.5B to 57% on GSM8K   ·   single L40S   ·   "
             "fastest training run wins", 29, GRAY),
        text(LEFT_X, 201,
             "track 1  ·  every record re-run 3× with fresh seeds on identical "
             "hardware  ·  free to attempt", 22, DIM),
    ]

    label_y0, pitch = ROW_TOP, row_pitch(len(rows))
    for i, rec in enumerate(rows):
        ly = label_y0 + i * pitch
        by = ly + 30
        is_rec = rec["id"] == record_id
        handle = f"@{rec['github']}"
        left = f"#{rec['id']}   {handle} · {card_label(rec)}"
        tstr, astr = fmt_time(rec["time_seconds_mean"]), f"{mean_acc(rec):.1%}"
        if is_rec:
            parts.append(text(LEFT_X, ly, left, 31, WHITE, weight="bold"))
            parts.append(text(RIGHT_X, ly,
                              f"CURRENT RECORD    {tstr}  ·  {astr}", 31, GREEN,
                              weight="bold", anchor="end"))
            parts.append(bar(by, rec["time_seconds_mean"] / slowest, GREEN))
        else:
            parts.append(text(LEFT_X, ly, left, 31, GRAY))
            parts.append(text(RIGHT_X, ly, f"{tstr}  ·  {astr}", 31, GRAY,
                              anchor="end"))
            parts.append(bar(by, rec["time_seconds_mean"] / slowest, BAR_GRAY))

    parts.append(text(LEFT_X, 812, f"Beat {fmt_time(fastest)}", 40, GREEN, weight="bold"))
    parts.append(text(LEFT_X + 250, 812,
                      "→   github.com/Saivineeth147/lora-speedrun", 33, GRAY))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    data = json.loads((REPO_ROOT / "records" / "records.json").read_text())
    svg = build_svg(data["records"])
    OUT_SVG.write_text(svg)
    subprocess.run(
        ["rsvg-convert", "-w", str(W), "-h", str(H), "-o", str(OUT_PNG), str(OUT_SVG)],
        check=True,
    )
    OUT_SVG.unlink()  # SVG is just an intermediate; only the PNG is committed
    print(f"wrote {OUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
