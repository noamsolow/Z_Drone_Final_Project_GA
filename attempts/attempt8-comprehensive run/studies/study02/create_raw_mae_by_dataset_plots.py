"""Create before-calibration raw prediction and MAE-vs-distance plots per real dataset."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts" / "reports"
PLOTS = ROOT / "artifacts" / "plots"
SOURCE = REPORTS / "final_calibrated_predictions.csv"


def text(x: float, y: float, value: str, size: int, color: str, *, bold: bool = False, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{"700" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def chart_point(x: float, y: float, rect: tuple[float, float, float, float], x_max: float, y_max: float) -> tuple[float, float]:
    left, top, width, height = rect
    return left + x / x_max * width, top + height - y / y_max * height


def line_path(values: list[tuple[float, float]], rect: tuple[float, float, float, float], x_max: float, y_max: float) -> str:
    points = [chart_point(x, y, rect, x_max, y_max) for x, y in values]
    return " ".join(("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points))


def build_plot(dataset: str, rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["true_distance_m"])].append(row)
    summary: list[dict[str, str]] = []
    for distance in sorted(grouped):
        group = grouped[distance]
        raw_predictions = [float(row["raw_predicted_distance_m"]) for row in group]
        mean_raw = sum(raw_predictions) / len(raw_predictions)
        mae = sum(abs(prediction - distance) for prediction in raw_predictions) / len(raw_predictions)
        summary.append({
            "dataset": dataset,
            "true_distance_m": f"{distance:.1f}",
            "count": str(len(group)),
            "mean_raw_prediction_m": f"{mean_raw:.4f}",
            "mae_m": f"{mae:.4f}",
        })

    distances = [float(row["true_distance_m"]) for row in summary]
    predictions = [float(row["mean_raw_prediction_m"]) for row in summary]
    errors = [float(row["mae_m"]) for row in summary]
    raw_mae = sum(abs(float(row["raw_predicted_distance_m"]) - float(row["true_distance_m"])) for row in rows) / len(rows)
    overprediction = sum(float(row["raw_predicted_distance_m"]) > float(row["true_distance_m"]) for row in rows) / len(rows)
    x_max = max(distances)
    y_top_max = max(max(predictions), x_max)
    y_top_max = (int(y_top_max / 10) + 1) * 10
    y_bottom_max = (int(max(errors) / 10) + 1) * 10
    top_rect = (100.0, 255.0, 1040.0, 240.0)
    bottom_rect = (100.0, 590.0, 1040.0, 160.0)
    navy, orange, green, muted = "#0b2463", "#ff8748", "#2f8a67", "#4b5563"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="820" viewBox="0 0 1280 820">',
        '<rect width="1280" height="820" fill="#e9edf5"/>',
        text(72, 66, "REAL-DOMAIN PERFORMANCE · BEFORE CALIBRATION", 15, orange, bold=True),
        text(72, 112, f"{dataset}: Raw Prediction and MAE by True Distance", 34, navy, bold=True),
        text(72, 145, "Raw RF-XGBoost ensemble predictions before the real-domain quadratic correction.", 17, muted),
        '<line x1="72" y1="170" x2="1208" y2="170" stroke="#d6dde9" stroke-width="2"/>',
        '<rect x="72" y="190" width="250" height="48" rx="10" fill="#0b2463"/>',
        text(197, 221, f"Raw MAE  {raw_mae:.2f} m", 17, "#ffffff", bold=True, anchor="middle"),
        '<rect x="340" y="190" width="300" height="48" rx="10" fill="#ff8748"/>',
        text(490, 221, f"Overprediction  {overprediction * 100:.1f}%", 17, "#ffffff", bold=True, anchor="middle"),
        text(100, 282, "Mean raw prediction vs. real depth", 19, navy, bold=True),
        text(100, 617, "MAE by real depth", 19, navy, bold=True),
    ]
    for rect, y_max in ((top_rect, y_top_max), (bottom_rect, y_bottom_max)):
        left, top, width, height = rect
        lines.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="8" fill="#ffffff"/>')
        for y_tick in range(0, int(y_max) + 1, 10):
            _, y = chart_point(0, y_tick, rect, x_max, y_max)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e6ebf3" stroke-width="1"/>')
            lines.append(text(left - 12, y + 4, str(y_tick), 12, muted, anchor="end"))
        for distance in distances:
            x, _ = chart_point(distance, 0, rect, x_max, y_max)
            lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="#f0f3f8" stroke-width="1"/>')
            lines.append(text(x, top + height + 23, f"{distance:g}", 12, muted, anchor="middle"))

    lines.extend([
        f'<path d="{line_path(list(zip(distances, distances)), top_rect, x_max, y_top_max)}" fill="none" stroke="{green}" stroke-width="4" stroke-dasharray="10 7"/>',
        f'<path d="{line_path(list(zip(distances, predictions)), top_rect, x_max, y_top_max)}" fill="none" stroke="{orange}" stroke-width="5"/>',
    ])
    for distance, prediction in zip(distances, predictions):
        x, y = chart_point(distance, prediction, top_rect, x_max, y_top_max)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{orange}"/>')
    bar_width = min(42.0, bottom_rect[2] / max(len(distances), 1) * 0.58)
    for distance, error in zip(distances, errors):
        x, y = chart_point(distance, error, bottom_rect, x_max, y_bottom_max)
        base_x, base_y = chart_point(distance, 0, bottom_rect, x_max, y_bottom_max)
        lines.append(f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{base_y - y:.1f}" rx="4" fill="{navy}"/>')
        lines.append(text(x, y - 7, f"{error:.1f}", 12, navy, bold=True, anchor="middle"))
    lines.extend([
        text(1168, 303, "Real depth", 14, green, bold=True),
        '<line x1="1168" y1="315" x2="1204" y2="315" stroke="#2f8a67" stroke-width="4" stroke-dasharray="8 5"/>',
        text(1168, 347, "Raw prediction", 14, orange, bold=True),
        '<line x1="1168" y1="359" x2="1204" y2="359" stroke="#ff8748" stroke-width="5"/>',
        text(620, 788, "True distance (m)", 16, navy, bold=True, anchor="middle"),
        '<text x="28" y="375" font-family="Aptos, Arial, sans-serif" font-size="16" font-weight="700" fill="#0b2463" transform="rotate(-90 28 375)" text-anchor="middle">Distance (m)</text>',
        '<text x="28" y="670" font-family="Aptos, Arial, sans-serif" font-size="16" font-weight="700" fill="#0b2463" transform="rotate(-90 28 670)" text-anchor="middle">MAE (m)</text>',
        '</svg>',
    ])
    return "\n".join(lines), summary


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_dataset[row["drone_type"]].append(row)
    all_rows: list[dict[str, str]] = []
    PLOTS.mkdir(parents=True, exist_ok=True)
    for dataset, rows in sorted(by_dataset.items()):
        svg, summary = build_plot(dataset, rows)
        slug = dataset.lower().replace(" ", "_")
        path = PLOTS / f"raw_{slug}_mae_vs_distance.svg"
        path.write_text(svg, encoding="utf-8")
        all_rows.extend(summary)
        print(f"Created {path}")
    output_csv = REPORTS / "raw_mae_by_distance_per_dataset.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["dataset", "true_distance_m", "count", "mean_raw_prediction_m", "mae_m"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Created {output_csv}")


if __name__ == "__main__":
    main()
