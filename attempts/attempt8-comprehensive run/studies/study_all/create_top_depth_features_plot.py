"""Render the saved Attempt 8 lower-RF feature importances as an editable SVG."""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts" / "reports"
PLOTS = ROOT / "artifacts" / "plots"
SOURCE = REPORTS / "lower_model_feature_importances.csv"
OUTPUT = PLOTS / "attempt8_top_depth_features.svg"
SUMMARY = REPORTS / "attempt8_top_depth_features.csv"

LABELS = {
    "bbox_only__bbox_midpoint__object_depth": "Drone box: midpoint depth",
    "bbox_only__inner50_median__object_depth": "Drone box: inner-50% median depth",
    "bbox_only__bbox_mean__object_depth": "Drone box: mean depth",
    "bbox_expand_1_5x__bbox_midpoint__object_depth": "1.5x context: midpoint depth",
    "bbox_expand_1_5x__inner50_median__object_depth": "1.5x context: inner-50% median depth",
    "bbox_expand_1_5x__bbox_mean__object_depth": "1.5x context: mean depth",
    "bbox_expand_2x__bbox_midpoint__object_depth": "2x context: midpoint depth",
    "bbox_expand_2x__inner50_median__object_depth": "2x context: inner-50% median depth",
    "bbox_expand_2x__bbox_mean__object_depth": "2x context: mean depth",
    "bbox_expand_4x__bbox_midpoint__object_depth": "4x context: midpoint depth",
    "bbox_expand_4x__inner50_median__object_depth": "4x context: inner-50% median depth",
    "bbox_expand_4x__bbox_mean__object_depth": "4x context: mean depth",
    "full_image__bbox_midpoint__object_depth": "Full image: midpoint depth",
    "full_image__inner50_median__object_depth": "Full image: inner-50% median depth",
    "full_image__bbox_mean__object_depth": "Full image: mean depth",
}


def svg_text(x: float, y: float, value: str, size: int, fill: str, *, weight: str = "normal", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rows.sort(key=lambda row: float(row["importance_value"]), reverse=True)
    rows = rows[:10]
    if len(rows) != 10:
        raise RuntimeError("Expected at least 10 saved feature-importance rows.")

    PLOTS.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["rank", "feature", "importance"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, "feature": LABELS.get(row["term"], row["term"]), "importance": f'{float(row["importance_value"]):.6f}'})

    max_value = max(float(row["importance_value"]) for row in rows)
    canvas_w, canvas_h = 1280, 720
    chart_x, chart_y, chart_w, row_h = 470, 192, 600, 43
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="1280" height="720" fill="#e9edf5"/>',
        svg_text(80, 72, "ATTEMPT 8 · FEATURE IMPORTANCE", 16, "#ff8748", weight="700"),
        svg_text(80, 122, "Top Depth Features", 40, "#0b2463", weight="700"),
        svg_text(80, 157, "Saved lower Random Forest component used in the improved-linear model", 19, "#4b5563"),
        '<line x1="80" y1="181" x2="1200" y2="181" stroke="#d6dde9" stroke-width="2"/>',
        svg_text(chart_x, 208, "Relative importance", 15, "#4b5563", weight="700"),
        svg_text(80, 675, "Higher values indicate stronger contribution within this depth-feature Random Forest.", 15, "#4b5563"),
    ]

    for index, row in enumerate(rows):
        y = chart_y + index * row_h
        value = float(row["importance_value"])
        label = LABELS.get(row["term"], row["term"])
        bar_width = (value / max_value) * chart_w
        fill = "#ff8748" if index == 0 else "#0b2463" if index < 3 else "#2e5e9e"
        lines.extend(
            [
                svg_text(100, y + 27, f"{index + 1:02d}", 15, "#8090a9", weight="700"),
                svg_text(145, y + 27, label, 17, "#111827", weight="700" if index < 3 else "normal"),
                f'<rect x="{chart_x}" y="{y}" width="{chart_w}" height="29" rx="6" fill="#d6dde9"/>',
                f'<rect x="{chart_x}" y="{y}" width="{bar_width:.1f}" height="29" rx="6" fill="{fill}"/>',
                svg_text(1105, y + 22, f"{value:.3f}", 17, fill, weight="700", anchor="end"),
            ]
        )
    lines.append("</svg>")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {OUTPUT}")
    print(f"Created {SUMMARY}")


if __name__ == "__main__":
    main()
