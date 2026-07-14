"""Create raw (uncalibrated) mean-prediction plots by real distance for each dataset."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts" / "reports" / "raw_mae_by_distance_per_dataset.csv"
PLOTS = ROOT / "artifacts" / "plots"


def text(x: float, y: float, value: str, size: int, color: str, *, bold: bool = False, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" font-weight="{"700" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'


def point(x: float, y: float, left: float, top: float, width: float, height: float, x_max: float, y_max: float) -> tuple[float, float]:
    return left + x / x_max * width, top + height - y / y_max * height


def path(values: list[tuple[float, float]], left: float, top: float, width: float, height: float, x_max: float, y_max: float) -> str:
    return " ".join(("M" if i == 0 else "L") + f"{px:.1f},{py:.1f}" for i, (px, py) in enumerate(point(x, y, left, top, width, height, x_max, y_max) for x, y in values))


def render(dataset: str, rows: list[dict[str, str]]) -> str:
    distances = [float(row["true_distance_m"]) for row in rows]
    raw = [float(row["mean_raw_prediction_m"]) for row in rows]
    x_max = max(distances)
    y_max = (int(max(max(raw), x_max) / 10) + 1) * 10
    left, top, width, height = 105.0, 75.0, 1060.0, 540.0
    black, raw_red, muted = "#151515", "#b04f3a", "#4b5563"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">',
        '<rect width="1280" height="720" fill="#ffffff"/>',
        text(640, 40, f"{dataset} Mean Predictions by Distance — Before Calibration", 24, "#222222", bold=True, anchor="middle"),
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#555555" stroke-width="2"/>',
    ]
    for tick in range(0, int(y_max) + 1, 20):
        _, y = point(0, tick, left, top, width, height, x_max, y_max)
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e7e7e7" stroke-width="1"/>', text(left - 15, y + 5, str(tick), 15, muted, anchor="end")]
    for distance in distances:
        x, _ = point(distance, 0, left, top, width, height, x_max, y_max)
        lines += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="#eeeeee" stroke-width="1"/>', text(x, top + height + 28, f"{distance:g}", 15, muted, anchor="middle")]
    lines += [
        f'<path d="{path(list(zip(distances, distances)), left, top, width, height, x_max, y_max)}" fill="none" stroke="{black}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<path d="{path(list(zip(distances, raw)), left, top, width, height, x_max, y_max)}" fill="none" stroke="{raw_red}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>',
    ]
    for distance, value in zip(distances, distances):
        x, y = point(distance, value, left, top, width, height, x_max, y_max)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{black}"/>')
    for distance, value in zip(distances, raw):
        x, y = point(distance, value, left, top, width, height, x_max, y_max)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{raw_red}"/>')
    lines += [
        '<line x1="135" y1="112" x2="185" y2="112" stroke="#151515" stroke-width="5"/>', text(202, 120, "True distance", 22, black, bold=True),
        '<line x1="135" y1="150" x2="185" y2="150" stroke="#b04f3a" stroke-width="6"/>', text(202, 158, "Raw ensemble prediction", 22, raw_red, bold=True),
        text(640, 685, "True distance bucket (m)", 17, "#333333", bold=True, anchor="middle"),
        '<text x="35" y="345" font-family="Aptos, Arial, sans-serif" font-size="17" font-weight="700" fill="#333333" transform="rotate(-90 35 345)" text-anchor="middle">Mean distance (m)</text>',
        '</svg>',
    ]
    return "\n".join(lines)


def main() -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            groups[row["dataset"]].append(row)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for dataset, rows in groups.items():
        rows.sort(key=lambda row: float(row["true_distance_m"]))
        output = PLOTS / f"raw_{dataset.lower()}_mean_predictions_by_distance.svg"
        output.write_text(render(dataset, rows), encoding="utf-8")
        print(f"Created {output}")


if __name__ == "__main__":
    main()
