"""Overwrite the raw per-dataset plots with charts only: no slide framing or metric cards."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts" / "reports"
PLOTS = ROOT / "artifacts" / "plots"
SOURCE = REPORTS / "raw_mae_by_distance_per_dataset.csv"


def text(x: float, y: float, value: str, size: int, color: str, *, bold: bool = False, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" font-weight="{"700" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'


def point(x: float, y: float, rect: tuple[float, float, float, float], x_max: float, y_max: float) -> tuple[float, float]:
    left, top, width, height = rect
    return left + x / x_max * width, top + height - y / y_max * height


def line(values: list[tuple[float, float]], rect: tuple[float, float, float, float], x_max: float, y_max: float) -> str:
    return " ".join(("M" if i == 0 else "L") + f"{px:.1f},{py:.1f}" for i, (px, py) in enumerate(point(x, y, rect, x_max, y_max) for x, y in values))


def make_chart(dataset: str, rows: list[dict[str, str]]) -> str:
    distances = [float(row["true_distance_m"]) for row in rows]
    errors = [float(row["mae_m"]) for row in rows]
    x_max = max(distances)
    # Keep the MAE axis compact to the actual error range, independent of depth.
    y_max = max(errors) * 1.08
    y_tick_step = 5 if y_max <= 45 else 10
    chart = (115.0, 55.0, 1045.0, 555.0)
    left, top, width, height = chart
    navy, muted = "#0b2463", "#4b5563"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="700" viewBox="0 0 1280 700">',
        '<rect width="1280" height="700" fill="#ffffff"/>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d6dde9" stroke-width="1"/>',
    ]
    for value in range(0, int(y_max) + 1, y_tick_step):
        _, y = point(0, value, chart, x_max, y_max)
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#edf0f5" stroke-width="1"/>', text(left - 12, y + 5, str(value), 14, muted, anchor="end")]
    for value in range(0, int(x_max) + 1, 10):
        _, y = point(0, value, chart, x_max, x_max)
        lines.append(text(left + width + 12, y + 5, str(value), 14, "#2f8a67"))
    for distance in distances:
        x, _ = point(distance, 0, chart, x_max, y_max)
        lines += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="#f4f6f9" stroke-width="1"/>', text(x, top + height + 27, f"{distance:g}", 14, muted, anchor="middle")]
    lines.append(f'<path d="{line(list(zip(distances, errors)), chart, x_max, y_max)}" fill="none" stroke="{navy}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>')
    # True depth uses the right axis so it can be compared visually without
    # distorting the compact MAE scale on the left axis.
    lines.append(f'<path d="{line(list(zip(distances, distances)), chart, x_max, x_max)}" fill="none" stroke="#2f8a67" stroke-width="4" stroke-dasharray="10 7"/>')
    for distance, error in zip(distances, errors):
        x, y = point(distance, error, chart, x_max, y_max)
        lines += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{navy}"/>', text(x, y - 12, f"{error:.1f}", 12, navy, bold=True, anchor="middle")]
    lines += [
        text(638, 670, "True depth (m)", 17, navy, bold=True, anchor="middle"),
        '<text x="35" y="333" font-family="Aptos, Arial, sans-serif" font-size="17" font-weight="700" fill="#0b2463" transform="rotate(-90 35 333)" text-anchor="middle">MAE (m)</text>',
        '<text x="1240" y="333" font-family="Aptos, Arial, sans-serif" font-size="17" font-weight="700" fill="#2f8a67" transform="rotate(-90 1240 333)" text-anchor="middle">True depth (m)</text>',
        '<line x1="955" y1="82" x2="993" y2="82" stroke="#2f8a67" stroke-width="4" stroke-dasharray="10 7"/>',
        text(1005, 87, "True depth", 14, "#2f8a67", bold=True),
        '</svg>',
    ]
    return "\n".join(lines)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as source:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in csv.DictReader(source):
            groups[row["dataset"]].append(row)
    for dataset, rows in groups.items():
        rows.sort(key=lambda row: float(row["true_distance_m"]))
        path = PLOTS / f"raw_{dataset.lower()}_mae_vs_distance.svg"
        path.write_text(make_chart(dataset, rows), encoding="utf-8")
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
