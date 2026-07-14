"""Create the Attempt 8 final-ensemble MAE table by weather and time of day."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts" / "reports"
PLOTS = ROOT / "artifacts" / "plots"
PREDICTIONS = REPORTS / "predictions.csv"
SUMMARY = REPORTS / "final_ensemble_weather_time_mae.csv"
SVG = PLOTS / "final_ensemble_weather_time_mae_heatmap.svg"

MODEL_NAME = "tuned_weight_blend"
# cv_oof covers the development set and test covers the held-out set: together
# they provide one prediction for every image in the complete 15,064-row set.
SPLIT_NAMES = {"cv_oof", "test"}
WEATHER_ORDER = ["clear_sky", "light_rain"]
TIME_ORDER = ["10AM", "8PM"]


def main() -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with PREDICTIONS.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["model_name"] != MODEL_NAME or row["split_name"] not in SPLIT_NAMES:
                continue
            grouped[(row["weather"], row["time_of_day"])].append(float(row["absolute_error_m"]))

    expected = {(weather, time) for weather in WEATHER_ORDER for time in TIME_ORDER}
    missing = expected - set(grouped)
    if missing:
        raise RuntimeError(f"Missing Attempt 8 predictions for: {sorted(missing)}")

    PLOTS.mkdir(parents=True, exist_ok=True)
    table = [[sum(grouped[(weather, time)]) / len(grouped[(weather, time)]) for time in TIME_ORDER] for weather in WEATHER_ORDER]

    with SUMMARY.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["model", "split", "weather", "time_of_day", "count", "mae_m"])
        for weather in WEATHER_ORDER:
            for time in TIME_ORDER:
                errors = grouped[(weather, time)]
                writer.writerow(["RF-XGBoost ensemble", "full dataset (OOF + test)", weather, time, len(errors), f"{sum(errors) / len(errors):.4f}"])

    flat = [value for row in table for value in row]
    low, high = min(flat), max(flat)

    def color(value: float) -> str:
        # Light blue to AirDepth navy; lower MAE is intentionally lighter.
        fraction = 0.5 if high == low else (value - low) / (high - low)
        start, end = (220, 232, 248), (11, 36, 99)
        rgb = [round(start[i] + (end[i] - start[i]) * fraction) for i in range(3)]
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def text(x: float, y: float, value: str, size: int, fill: str, weight: str = "normal", anchor: str = "start") -> str:
        return f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{value}</text>'

    canvas_w, canvas_h = 1280, 720
    grid_x, grid_y, cell_w, cell_h = 350, 235, 350, 170
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="1280" height="720" fill="#e9edf5"/>',
        text(80, 78, "FINAL ENSEMBLE", 16, "#ff8748", "700"),
        text(80, 128, "MAE by Weather and Time of Day", 38, "#0b2463", "700"),
        text(80, 162, "Attempt 8 — complete 15,064-image synthetic dataset (OOF + held-out test) | lower is better", 19, "#4b5563"),
        '<line x1="80" y1="190" x2="1200" y2="190" stroke="#d6dde9" stroke-width="2"/>',
        text(grid_x + cell_w / 2, 212, "10 AM", 20, "#0b2463", "700", "middle"),
        text(grid_x + cell_w + cell_w / 2, 212, "8 PM", 20, "#0b2463", "700", "middle"),
        text(80, grid_y + 90, "Clear sky", 21, "#0b2463", "700"),
        text(80, grid_y + cell_h + 90, "Light rain", 21, "#0b2463", "700"),
        text(grid_x + cell_w, 650, "Time of day", 18, "#0b2463", "700", "middle"),
        text(80, 675, "Each cell: mean absolute error in meters", 15, "#4b5563"),
    ]
    for row_index, weather in enumerate(WEATHER_ORDER):
        for col_index, time in enumerate(TIME_ORDER):
            value = table[row_index][col_index]
            x, y = grid_x + col_index * cell_w, grid_y + row_index * cell_h
            fill = color(value)
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="10" fill="{fill}"/>')
            lines.append(text(x + (cell_w - 8) / 2, y + 75, f"{value:.2f} m", 32, "#ffffff", "700", "middle"))
            lines.append(text(x + (cell_w - 8) / 2, y + 109, f"n = {len(grouped[(weather, time)])}", 15, "#ffffff", "normal", "middle"))
    lines.append("</svg>")
    SVG.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {SVG}")
    print(f"Created {SUMMARY}")


if __name__ == "__main__":
    main()
