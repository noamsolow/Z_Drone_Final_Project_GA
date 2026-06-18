from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS_ROOT = REPO_ROOT / "attempts" / "attempt8-comprehensive run" / "studies" / "study_all" / "artifacts"
REPORTS_DIR = ARTIFACTS_ROOT / "reports"
PLOTS_DIR = ARTIFACTS_ROOT / "plots"

MODEL_ORDER = [
    "scale only",
    "depth linear",
    "linear+bb",
    "improved linear",
    "RF",
    "RF+jitter",
    "XGBoost+jitter",
    "ensemble",
]

MODEL_COLORS = {
    "scale only": "#9ca3af",
    "depth linear": "#64748b",
    "linear+bb": "#475569",
    "improved linear": "#1f2937",
    "RF": "#2563eb",
    "RF+jitter": "#0f766e",
    "XGBoost+jitter": "#d97706",
    "ensemble": "#dc2626",
}

MODEL_DESCRIPTIONS = {
    "scale only": "A minimal calibration baseline that learns a global scale from the raw depth signal.",
    "depth linear": "A linear model using relative depth as the main predictor.",
    "linear+bb": "A linear model that adds bbox geometry and image metadata to the depth signal.",
    "improved linear": "A stacked linear model that adds a lower-stage RF prediction as an extra feature.",
    "RF": "A random forest trained on exact depth, bbox geometry, and metadata features.",
    "RF+jitter": "A random forest trained on aggregated jitter-summary features for more robust bbox/depth signals.",
    "XGBoost+jitter": "A boosted-tree model trained on the same aggregated jitter-summary feature table.",
    "ensemble": "A tuned blend of RF+jitter and XGBoost+jitter; the final best poster model.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create poster SVG plots from study_all artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=ARTIFACTS_ROOT,
        help="Root artifacts directory containing reports/ and audit/.",
    )
    return parser.parse_args()


def configure_artifact_paths(artifacts_root: Path) -> None:
    global ARTIFACTS_ROOT, REPORTS_DIR, PLOTS_DIR
    ARTIFACTS_ROOT = artifacts_root.resolve()
    REPORTS_DIR = ARTIFACTS_ROOT / "reports"
    PLOTS_DIR = ARTIFACTS_ROOT / "plots"


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [coerce_row(dict(row)) for row in csv.DictReader(handle)]


def coerce_row(row: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        stripped = "" if value is None else value.strip()
        if stripped == "":
            output[key] = value
            continue
        try:
            output[key] = float(stripped)
        except ValueError:
            output[key] = value
    return output


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    fill: str = "#111827",
    *,
    weight: int = 400,
    anchor: str = "start",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform_attr}>{escape(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    *,
    rx: float = 0,
    stroke: str | None = None,
    stroke_width: float = 1.0,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" fill="{fill}"{stroke_attr}/>'


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    *,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def circle(cx: float, cy: float, radius: float, fill: str, *, stroke: str = "#ffffff") -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def polyline(points: Sequence[tuple[float, float]], color: str, *, width: float = 2.4) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def save_svg(path: Path, width: int, height: int, parts: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    content.extend(parts)
    content.append("</svg>")
    path.write_text("\n".join(content), encoding="utf-8")


def model_sort_key(name: str) -> int:
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER)


def axis_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1:
        return [min_value]
    return [min_value + (max_value - min_value) * idx / (count - 1) for idx in range(count)]


def make_line_chart(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    title: str,
    subtitle: str,
    sample_context: str,
    y_label: str,
    output_path: Path,
    y_max: float | None = None,
    y_as_percent: bool = False,
    base_line_width: float = 2.0,
    highlight_line_width: float = 2.6,
    base_point_radius: float = 3.2,
    highlight_point_radius: float = 4.0,
) -> None:
    width = 1480
    height = 880
    left = 132
    right = 280
    top = 142
    bottom = 105
    chart_w = width - left - right
    chart_h = height - top - bottom

    test_rows = [row for row in rows if str(row["split_name"]) == "test"]
    distances = sorted({float(row["true_distance_m"]) for row in test_rows})
    values = [float(row[value_key]) for row in test_rows]
    min_x = min(distances)
    max_x = max(distances)
    min_y = 0.0
    max_y = y_max if y_max is not None else max(values) * 1.08

    def x_scale(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * chart_w

    def y_scale(value: float) -> float:
        clipped = min(max(value, min_y), max_y)
        return top + (max_y - clipped) / (max_y - min_y) * chart_h

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 58, title, 34, weight=700),
        text(left, 88, subtitle, 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
        text(left + chart_w / 2, height - 24, "True distance (m)", 20, "#374151", weight=700, anchor="middle"),
        text(
            34,
            top + chart_h / 2,
            y_label,
            20,
            "#374151",
            weight=700,
            anchor="middle",
            transform=f"rotate(-90 34 {top + chart_h / 2})",
        ),
    ]

    for tick in axis_ticks(min_y, max_y, 7):
        y = y_scale(tick)
        parts.append(line(left, y, left + chart_w, y, "#e5e7eb", dash="5 7"))
        label = pct(tick, 0) if y_as_percent else fmt(tick, 0)
        parts.append(text(left - 16, y + 6, label, 16, "#6b7280", anchor="end"))

    for tick in distances:
        x = x_scale(tick)
        parts.append(line(x, top, x, top + chart_h, "#f1f5f9"))
        parts.append(text(x, top + chart_h + 32, fmt(tick, 0), 16, "#6b7280", anchor="middle"))

    parts.append(line(left, top + chart_h, left + chart_w, top + chart_h, "#111827", width=1.5))
    parts.append(line(left, top, left, top + chart_h, "#111827", width=1.5))

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in test_rows:
        by_model.setdefault(str(row["display_name"]), []).append(row)

    for name in sorted(by_model, key=model_sort_key):
        model_rows = sorted(by_model[name], key=lambda row: float(row["true_distance_m"]))
        color = MODEL_COLORS.get(name, "#64748b")
        points = [(x_scale(float(row["true_distance_m"])), y_scale(float(row[value_key]))) for row in model_rows]
        parts.append(polyline(points, color, width=highlight_line_width if name in {"ensemble", "XGBoost+jitter"} else base_line_width))
        for x, y in points:
            parts.append(circle(x, y, highlight_point_radius if name in {"ensemble", "XGBoost+jitter"} else base_point_radius, color))

    legend_x = left + chart_w + 42
    legend_y = top + 18
    parts.append(text(legend_x, legend_y - 28, "Models", 20, weight=700))
    for idx, name in enumerate(sorted(by_model, key=model_sort_key)):
        y = legend_y + idx * 34
        color = MODEL_COLORS.get(name, "#64748b")
        parts.append(line(legend_x, y, legend_x + 36, y, color, width=3))
        parts.append(circle(legend_x + 18, y, 4, color))
        parts.append(text(legend_x + 50, y + 6, name, 17, "#111827", weight=600))

    save_svg(output_path, width, height, parts)


def make_mae_ranking(
    test_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    sample_context: str,
) -> None:
    rows = sorted(test_rows, key=lambda row: float(row["mae"]), reverse=True)
    width = 1360
    height = 175 + len(rows) * 78
    left = 270
    right = 150
    top = 145
    row_h = 70
    bar_h = 34
    chart_w = width - left - right
    max_value = max(float(row["mae"]) for row in rows) * 1.08

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 56, "Test Error Across Models", 34, weight=700),
        text(left, 86, "Lower bars mean better distance estimates; the weakest baselines are shown first.", 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
    ]

    chart_bottom = top + len(rows) * row_h
    for tick in axis_ticks(0.0, max_value, 7):
        x = left + (tick / max_value) * chart_w
        parts.append(line(x, top - 8, x, chart_bottom + 8, "#e5e7eb", dash="5 7"))
        parts.append(text(x, chart_bottom + 34, fmt(tick, 0), 17, "#6b7280", anchor="middle"))

    for idx, row in enumerate(rows):
        name = str(row["display_name"])
        value = float(row["mae"])
        y = top + idx * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        bar_w = (value / max_value) * chart_w
        color = MODEL_COLORS.get(name, "#64748b")
        parts.append(text(left - 18, cy + 7, name, 23, "#111827", weight=700, anchor="end"))
        parts.append(rect(left, y, chart_w, bar_h, "#f8fafc", rx=8))
        parts.append(rect(left, y, max(2.0, bar_w), bar_h, color, rx=8))
        parts.append(text(left + bar_w + 12, cy + 7, f"{fmt(value, 3)} m", 20, "#111827", weight=700))

    save_svg(output_path, width, height, parts)


def make_within10_ranking(
    test_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    sample_context: str,
) -> None:
    rows = sorted(test_rows, key=lambda row: float(row["within_10m_rate"]))
    width = 1360
    height = 175 + len(rows) * 78
    left = 270
    right = 140
    top = 145
    row_h = 70
    bar_h = 34
    chart_w = width - left - right
    max_value = 1.0

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 56, "Within 10m Accuracy: Worst to Best", 34, weight=700),
        text(left, 86, "Share of test images with <=10m absolute error; higher bars are better.", 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
    ]
    chart_bottom = top + len(rows) * row_h
    for tick in axis_ticks(0.0, max_value, 6):
        x = left + tick * chart_w
        parts.append(line(x, top - 8, x, chart_bottom + 8, "#e5e7eb", dash="5 7"))
        parts.append(text(x, chart_bottom + 34, pct(tick, 0), 17, "#6b7280", anchor="middle"))

    for idx, row in enumerate(rows):
        name = str(row["display_name"])
        value = float(row["within_10m_rate"])
        y = top + idx * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        bar_w = value * chart_w
        color = MODEL_COLORS.get(name, "#64748b")
        parts.append(text(left - 18, cy + 7, name, 23, "#111827", weight=700, anchor="end"))
        parts.append(rect(left, y, chart_w, bar_h, "#f8fafc", rx=8))
        parts.append(rect(left, y, max(2.0, bar_w), bar_h, color, rx=8))
        parts.append(text(left + bar_w + 12, cy + 7, pct(value, 1), 20, "#111827", weight=700))

    save_svg(output_path, width, height, parts)


def make_mean_relative_error_bar_chart(
    test_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    sample_context: str,
) -> None:
    rows = sorted(test_rows, key=lambda row: model_sort_key(str(row["display_name"])))
    width = 1480
    height = 900
    left = 120
    right = 80
    top = 145
    bottom = 170
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(float(row["mean_relative_error"]) for row in rows) * 1.10
    bar_w = min(108.0, chart_w / max(len(rows), 1) * 0.62)
    slot_w = chart_w / max(len(rows), 1)

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 56, "Mean Relative Error Across Major Models", 34, weight=700),
        text(left, 86, "Upward bars show normalized test error, making near/far performance easier to compare.", 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
        text(left + chart_w / 2, height - 24, "Major models", 20, "#374151", weight=700, anchor="middle"),
        text(
            34,
            top + chart_h / 2,
            "Mean relative error",
            20,
            "#374151",
            weight=700,
            anchor="middle",
            transform=f"rotate(-90 34 {top + chart_h / 2})",
        ),
    ]

    for tick in axis_ticks(0.0, max_value, 7):
        y = top + chart_h - chart_h * (tick / max_value)
        parts.append(line(left, y, left + chart_w, y, "#e5e7eb", dash="5 7"))
        parts.append(text(left - 16, y + 6, pct(tick, 0), 16, "#6b7280", anchor="end"))

    baseline_y = top + chart_h
    parts.append(line(left, baseline_y, left + chart_w, baseline_y, "#111827", width=1.5))
    parts.append(line(left, top, left, baseline_y, "#111827", width=1.5))

    for idx, row in enumerate(rows):
        name = str(row["display_name"])
        value = float(row["mean_relative_error"])
        cx = left + slot_w * (idx + 0.5)
        bar_h = chart_h * (value / max_value)
        x = cx - bar_w / 2
        y = baseline_y - bar_h
        color = MODEL_COLORS.get(name, "#64748b")
        parts.append(rect(x, y, bar_w, bar_h, color, rx=10))
        parts.append(text(cx, y - 12, pct(value, 1), 18, "#111827", weight=700, anchor="middle"))
        parts.append(
            text(
                cx,
                baseline_y + 28,
                name,
                18,
                "#111827",
                weight=700 if name == "ensemble" else 600,
                anchor="end",
                transform=f"rotate(-28 {cx} {baseline_y + 28})",
            )
        )

    save_svg(output_path, width, height, parts)


def make_best_model_distance_mae_bars(
    model_metrics: Sequence[dict[str, Any]],
    true_distance_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    sample_context: str,
) -> None:
    test_metric_rows = [row for row in model_metrics if str(row["split_name"]) == "test" and str(row["role"]) == "poster"]
    best_row = min(test_metric_rows, key=lambda row: float(row["mae"]))
    best_name = str(best_row["display_name"])
    overall_mae = float(best_row["mae"])
    rows = [
        row for row in true_distance_rows
        if str(row["split_name"]) == "test" and str(row["display_name"]) == best_name
    ]
    rows.sort(key=lambda row: float(row["true_distance_m"]))

    width = 1480
    height = 900
    left = 120
    right = 80
    top = 145
    bottom = 130
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(max(float(row["mae"]) for row in rows), overall_mae) * 1.12
    slot_w = chart_w / max(len(rows), 1)
    bar_w = min(74.0, slot_w * 0.68)

    band_colors = {
        "near": "#0f766e",
        "mid": "#0284c7",
        "far": "#d97706",
    }

    def distance_band(distance_value: float) -> str:
        if distance_value <= 50.0:
            return "near"
        if distance_value <= 90.0:
            return "mid"
        return "far"

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 56, f"{best_name} MAE by True Distance", 34, weight=700),
        text(left, 86, "Standing bars show where the winning model still gets harder as distance grows.", 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
        text(left + chart_w / 2, height - 24, "True distance (m)", 20, "#374151", weight=700, anchor="middle"),
        text(
            34,
            top + chart_h / 2,
            "MAE (m)",
            20,
            "#374151",
            weight=700,
            anchor="middle",
            transform=f"rotate(-90 34 {top + chart_h / 2})",
        ),
    ]

    for tick in axis_ticks(0.0, max_value, 7):
        y = top + chart_h - chart_h * (tick / max_value)
        parts.append(line(left, y, left + chart_w, y, "#e5e7eb", dash="5 7"))
        parts.append(text(left - 16, y + 6, fmt(tick, 0), 16, "#6b7280", anchor="end"))

    overall_y = top + chart_h - chart_h * (overall_mae / max_value)
    parts.append(line(left, overall_y, left + chart_w, overall_y, "#111827", width=2.2, dash="10 8"))
    parts.append(text(left + chart_w - 8, overall_y - 10, f"overall test MAE {fmt(overall_mae, 2)}m", 16, "#111827", weight=700, anchor="end"))

    baseline_y = top + chart_h
    parts.append(line(left, baseline_y, left + chart_w, baseline_y, "#111827", width=1.5))
    parts.append(line(left, top, left, baseline_y, "#111827", width=1.5))

    for idx, row in enumerate(rows):
        distance_value = float(row["true_distance_m"])
        mae_value = float(row["mae"])
        band = distance_band(distance_value)
        color = band_colors[band]
        cx = left + slot_w * (idx + 0.5)
        bar_h = chart_h * (mae_value / max_value)
        x = cx - bar_w / 2
        y = baseline_y - bar_h
        parts.append(rect(x, y, bar_w, bar_h, color, rx=10))
        parts.append(text(cx, y - 10, fmt(mae_value, 1), 16, "#111827", weight=700, anchor="middle"))
        parts.append(text(cx, baseline_y + 30, fmt(distance_value, 0), 17, "#111827", weight=600, anchor="middle"))

    legend_x = width - 280
    legend_y = 154
    parts.append(text(legend_x, legend_y - 22, "Distance bands", 18, "#111827", weight=700))
    for idx, band in enumerate(["near", "mid", "far"]):
        y = legend_y + idx * 28
        parts.append(rect(legend_x, y - 12, 18, 18, band_colors[band], rx=4))
        parts.append(text(legend_x + 28, y + 2, band, 17, "#111827", weight=600))

    save_svg(output_path, width, height, parts)


def make_distance_band_heatmap(
    distance_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    sample_context: str,
) -> None:
    test_rows = [row for row in distance_rows if str(row["split_name"]) == "test"]
    models = sorted({str(row["display_name"]) for row in test_rows}, key=model_sort_key)
    bands = ["near", "mid", "far"]
    lookup = {(str(row["display_name"]), str(row["distance_range"])): row for row in test_rows}
    values = [float(row["mae"]) for row in test_rows]
    min_v = min(values)
    max_v = max(values)

    def color(value: float) -> str:
        ratio = (value - min_v) / max(max_v - min_v, 1e-9)
        # Green to amber to red.
        if ratio < 0.5:
            local = ratio / 0.5
            r = int(22 + local * (245 - 22))
            g = int(163 + local * (158 - 163))
            b = int(74 + local * (11 - 74))
        else:
            local = (ratio - 0.5) / 0.5
            r = int(245 + local * (220 - 245))
            g = int(158 + local * (38 - 158))
            b = int(11 + local * (38 - 11))
        return f"#{r:02x}{g:02x}{b:02x}"

    width = 1080
    height = 845
    left = 260
    top = 175
    cell_w = 210
    cell_h = 66
    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 58, "MAE by Distance Band", 34, weight=700),
        text(left, 88, "Test MAE split into near, mid, and far ranges to reveal where each model fails.", 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
    ]

    for col, band in enumerate(bands):
        parts.append(text(left + col * cell_w + cell_w / 2, top - 24, band, 22, "#111827", weight=700, anchor="middle"))

    for row_idx, model in enumerate(models):
        y = top + row_idx * cell_h
        parts.append(text(left - 24, y + 42, model, 22, "#111827", weight=700, anchor="end"))
        for col, band in enumerate(bands):
            x = left + col * cell_w
            value = float(lookup[(model, band)]["mae"])
            fill = color(value)
            parts.append(rect(x, y, cell_w - 10, cell_h - 10, fill, rx=10))
            parts.append(text(x + (cell_w - 10) / 2, y + 38, f"{fmt(value, 2)}m", 21, "#ffffff", weight=700, anchor="middle"))

    save_svg(output_path, width, height, parts)


def make_cv_test_gap(
    model_metrics: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    sample_context: str,
) -> None:
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in model_metrics:
        by_name.setdefault(str(row["display_name"]), {})[str(row["split_name"])] = row
    rows = []
    for name, split_rows in by_name.items():
        rows.append(
            {
                "display_name": name,
                "gap": float(split_rows["test"]["mae"]) - float(split_rows["cv_oof"]["mae"]),
                "test_mae": float(split_rows["test"]["mae"]),
            }
        )
    rows.sort(key=lambda row: row["gap"])

    width = 1360
    height = 175 + len(rows) * 78
    left = 330
    right = 150
    top = 145
    row_h = 70
    bar_h = 34
    chart_w = width - left - right
    min_gap = min(row["gap"] for row in rows)
    max_gap = max(row["gap"] for row in rows)
    pad = max(abs(min_gap), abs(max_gap), 0.1) * 1.22
    min_x = -pad
    max_x = pad
    zero_x = left + (-min_x) / (max_x - min_x) * chart_w

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 56, "Generalization Gap: Test MAE - CV MAE", 34, weight=700),
        text(left, 86, "Values near zero mean the holdout test mirrors cross-validation.", 18, "#6b7280"),
        text(left, 112, sample_context, 17, "#64748b", weight=700),
        line(zero_x, top - 12, zero_x, top + len(rows) * row_h + 12, "#111827", width=2),
    ]
    for tick in axis_ticks(min_x, max_x, 7):
        x = left + (tick - min_x) / (max_x - min_x) * chart_w
        parts.append(line(x, top - 8, x, top + len(rows) * row_h + 8, "#e5e7eb", dash="5 7"))
        parts.append(text(x, top + len(rows) * row_h + 34, fmt(tick, 2), 17, "#6b7280", anchor="middle"))

    for idx, row in enumerate(rows):
        name = str(row["display_name"])
        gap = float(row["gap"])
        y = top + idx * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        x = left + (gap - min_x) / (max_x - min_x) * chart_w
        start = min(zero_x, x)
        bar_w = max(abs(x - zero_x), 2.0)
        color = MODEL_COLORS.get(name, "#64748b")
        parts.append(text(left - 18, cy + 7, name, 23, "#111827", weight=700, anchor="end"))
        parts.append(rect(start, y, bar_w, bar_h, color, rx=8))
        anchor = "start" if gap >= 0 else "end"
        label_x = x + 12 if gap >= 0 else x - 12
        parts.append(text(label_x, cy + 7, f"{gap:+.3f}m", 20, "#111827", weight=700, anchor=anchor))

    save_svg(output_path, width, height, parts)


def write_model_table(model_metrics: Sequence[dict[str, Any]]) -> None:
    test_rows = [row for row in model_metrics if str(row["split_name"]) == "test"]
    test_rows.sort(key=lambda row: float(row["mae"]))

    csv_path = REPORTS_DIR / "poster_model_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["rank", "model", "role", "description", "test_mae", "test_r2", "within_10m_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(test_rows, start=1):
            name = str(row["display_name"])
            writer.writerow(
                {
                    "rank": rank,
                    "model": name,
                    "role": str(row["role"]),
                    "description": MODEL_DESCRIPTIONS.get(name, ""),
                    "test_mae": fmt(float(row["mae"]), 3),
                    "test_r2": fmt(float(row["r2"]), 3),
                    "within_10m_rate": pct(float(row["within_10m_rate"]), 1),
                }
            )

    md_lines = [
        "# Poster Model Table",
        "",
        "| Rank | Model | Short explanation | Test MAE |",
        "| ---: | --- | --- | ---: |",
    ]
    for rank, row in enumerate(test_rows, start=1):
        name = str(row["display_name"])
        md_lines.append(
            f"| {rank} | {name} | {MODEL_DESCRIPTIONS.get(name, '')} | {fmt(float(row['mae']), 3)}m |"
        )
    (REPORTS_DIR / "poster_model_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def delete_old_plots() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in PLOTS_DIR.glob("*.svg"):
        path.unlink()


def main() -> None:
    args = parse_args()
    configure_artifact_paths(args.artifacts_root)

    model_metrics = read_csv_rows(REPORTS_DIR / "model_metrics.csv")
    true_distance_summary = read_csv_rows(REPORTS_DIR / "true_distance_summary.csv")
    distance_range_metrics = read_csv_rows(REPORTS_DIR / "distance_range_metrics.csv")
    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    coverage = json.loads((ARTIFACTS_ROOT / "audit" / "coverage_summary.json").read_text(encoding="utf-8"))

    delete_old_plots()

    test_rows = [row for row in model_metrics if str(row["split_name"]) == "test"]
    sample_context = (
        f"Total n={int(coverage['num_common_rows']):,} | "
        f"Train/CV n={int(coverage['num_dev_rows']):,} | "
        f"Test n={int(coverage['num_test_rows']):,} | "
        f"Dropped={int(coverage['num_dropped_rows'])}"
    )
    make_line_chart(
        true_distance_summary,
        value_key="mae",
        title="MAE by True Distance",
        subtitle="Per-distance test error shows where each model breaks down.",
        sample_context=sample_context,
        y_label="MAE (m)",
        output_path=PLOTS_DIR / "poster_all_models_mae_by_true_distance.svg",
    )
    make_line_chart(
        true_distance_summary,
        value_key="mae",
        title="MAE by True Distance (Zoomed)",
        subtitle="Same test-distance view, but zoomed into the stronger-error range with thicker model traces.",
        sample_context=sample_context,
        y_label="MAE (m)",
        output_path=PLOTS_DIR / "poster_all_models_mae_by_true_distance_zoomed.svg",
        y_max=35.0,
        base_line_width=3.0,
        highlight_line_width=4.4,
        base_point_radius=4.2,
        highlight_point_radius=5.4,
    )
    make_line_chart(
        true_distance_summary,
        value_key="mean_relative_error",
        title="Relative Error by True Distance",
        subtitle="Normalizing by distance highlights near-range instability and far-range bias.",
        sample_context=sample_context,
        y_label="Mean relative error",
        output_path=PLOTS_DIR / "poster_all_models_relative_error_by_true_distance.svg",
        y_as_percent=True,
    )
    make_best_model_distance_mae_bars(
        model_metrics,
        true_distance_summary,
        PLOTS_DIR / "poster_best_model_mae_by_true_distance_bars.svg",
        sample_context=sample_context,
    )
    make_mae_ranking(
        test_rows,
        PLOTS_DIR / "poster_test_mae_worst_to_best.svg",
        sample_context=sample_context,
    )
    make_within10_ranking(
        test_rows,
        PLOTS_DIR / "poster_within_10m_worst_to_best.svg",
        sample_context=sample_context,
    )
    make_mean_relative_error_bar_chart(
        test_rows,
        PLOTS_DIR / "poster_mean_relative_error_major_models.svg",
        sample_context=sample_context,
    )
    make_distance_band_heatmap(
        distance_range_metrics,
        PLOTS_DIR / "poster_distance_band_mae_heatmap.svg",
        sample_context=sample_context,
    )
    make_cv_test_gap(
        model_metrics,
        PLOTS_DIR / "poster_cv_test_generalization_gap.svg",
        sample_context=sample_context,
    )
    write_model_table(model_metrics)

    manifest = {
        "source_summary_json": str(REPORTS_DIR / "summary.json"),
        "num_common_rows": int(coverage["num_common_rows"]),
        "num_dev_rows": int(coverage["num_dev_rows"]),
        "num_test_rows": int(coverage["num_test_rows"]),
        "num_dropped_rows": int(coverage["num_dropped_rows"]),
        "best_model": summary["poster_test_ranking"][0]["display_name"],
        "best_test_mae": summary["poster_test_ranking"][0]["mae"],
        "plots": sorted(path.name for path in PLOTS_DIR.glob("*.svg")),
        "table_csv": str(REPORTS_DIR / "poster_model_table.csv"),
        "table_md": str(REPORTS_DIR / "poster_model_table.md"),
    }
    (REPORTS_DIR / "poster_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
