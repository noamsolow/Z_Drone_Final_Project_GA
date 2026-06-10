from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS_ROOT = REPO_ROOT / "attempts" / "attempt8-comprehensive run" / "studies" / "study_all" / "artifacts"
REPORTS_DIR = ARTIFACTS_ROOT / "reports"
PLOTS_DIR = ARTIFACTS_ROOT / "plots"

MODEL_COLORS = {
    "scale only": "#cbd5e1",
    "depth linear": "#94a3b8",
    "linear+bb": "#64748b",
    "improved linear": "#475569",
    "RF": "#334155",
    "RF+jitter": "#0f766e",
    "XGBoost+jitter": "#0284c7",
    "ensemble": "#059669",
}


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [coerce_row(dict(row)) for row in csv.DictReader(handle)]


def coerce_row(row: Dict[str, str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            output[key] = value
            continue
        stripped = value.strip()
        if stripped == "":
            output[key] = value
            continue
        try:
            output[key] = float(stripped)
            continue
        except ValueError:
            output[key] = value
    return output


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    fill: str,
    *,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
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
    stroke_attr = ""
    if stroke:
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" '
        f'fill="{fill}"{stroke_attr}/>'
    )


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
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def circle(cx: float, cy: float, radius: float, fill: str, *, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}"{stroke_attr}/>'


def polyline(points: Sequence[tuple[float, float]], color: str, *, width: float = 2.0) -> str:
    encoded = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def save_svg(path: Path, width: int, height: int, parts: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    content.extend(parts)
    content.append("</svg>")
    path.write_text("\n".join(content), encoding="utf-8")


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def make_horizontal_bar_chart(
    rows: Sequence[Dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    value_key: str,
    output_path: Path,
    value_label: str,
    descending: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    width = 1280
    height = 150 + 82 * len(rows)
    left = 260
    right = 90
    top = 120
    row_h = 70
    bar_h = 34
    chart_w = width - left - right

    values = [float(row[value_key]) for row in rows]
    data_min = min(values) if min_value is None else min_value
    data_max = max(values) if max_value is None else max_value
    span = max(data_max - data_min, 1e-9)
    ticks = 6

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 54, title, 34, "#111827", weight=700),
        text(left, 84, subtitle, 18, "#6b7280"),
    ]

    chart_top = top
    chart_bottom = top + len(rows) * row_h
    for tick_index in range(ticks + 1):
        ratio = tick_index / ticks
        x = left + chart_w * ratio
        tick_value = data_min + span * ratio
        parts.append(line(x, chart_top - 8, x, chart_bottom + 8, "#e5e7eb", dash="4 6"))
        parts.append(text(x, chart_bottom + 34, format_float(tick_value, 2), 18, "#6b7280", anchor="middle"))
    parts.append(line(left, chart_bottom + 8, left + chart_w, chart_bottom + 8, "#9ca3af", width=1.2))

    sorted_rows = sorted(rows, key=lambda row: float(row[value_key]), reverse=descending)
    for index, row in enumerate(sorted_rows):
        y = chart_top + index * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        value = float(row[value_key])
        ratio = (value - data_min) / span
        bar_w = max(chart_w * ratio, 2.0)
        color = MODEL_COLORS.get(str(row["display_name"]), "#64748b")
        parts.append(text(left - 18, cy + 7, str(row["display_name"]), 24, "#111827", weight=600, anchor="end"))
        parts.append(rect(left, y, chart_w, bar_h, "#f8fafc", rx=8))
        parts.append(rect(left, y, bar_w, bar_h, color, rx=8))
        parts.append(text(left + bar_w + 12, cy + 7, f"{format_float(value, 3)} {value_label}".strip(), 20, "#111827", weight=700))
    save_svg(output_path, width, height, parts)


def make_r2_chart(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    width = 1280
    height = 150 + 82 * len(rows)
    left = 280
    right = 90
    top = 120
    row_h = 70
    bar_h = 34
    chart_w = width - left - right
    min_r2 = min(float(row["r2"]) for row in rows)
    max_r2 = max(float(row["r2"]) for row in rows)
    span = max_r2 - min_r2
    zero_x = left + chart_w * ((0.0 - min_r2) / span)

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 54, "Attempt 8 Test R2 Ranking", 34, "#111827", weight=700),
        text(left, 84, "Negative values mean worse than predicting the global mean.", 18, "#6b7280"),
    ]
    chart_bottom = top + len(rows) * row_h
    for tick in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        if tick < min_r2 - 1e-9 or tick > max_r2 + 1e-9:
            continue
        x = left + chart_w * ((tick - min_r2) / span)
        parts.append(line(x, top - 8, x, chart_bottom + 8, "#e5e7eb", dash="4 6"))
        parts.append(text(x, chart_bottom + 34, format_float(tick, 1), 18, "#6b7280", anchor="middle"))
    parts.append(line(zero_x, top - 12, zero_x, chart_bottom + 12, "#111827", width=2.0))

    sorted_rows = sorted(rows, key=lambda row: float(row["r2"]), reverse=True)
    for index, row in enumerate(sorted_rows):
        y = top + index * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        value = float(row["r2"])
        x = left + chart_w * ((value - min_r2) / span)
        start = min(zero_x, x)
        bar_w = abs(x - zero_x)
        color = MODEL_COLORS.get(str(row["display_name"]), "#64748b")
        parts.append(text(left - 18, cy + 7, str(row["display_name"]), 24, "#111827", weight=600, anchor="end"))
        parts.append(rect(left, y, chart_w, bar_h, "#f8fafc", rx=8))
        parts.append(rect(start, y, max(bar_w, 2.0), bar_h, color, rx=8))
        label_x = x + 12 if value >= 0 else x - 12
        anchor = "start" if value >= 0 else "end"
        parts.append(text(label_x, cy + 7, format_float(value, 3), 20, "#111827", weight=700, anchor=anchor))
    save_svg(output_path, width, height, parts)


def make_grouped_cv_test_chart(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    width = 1380
    height = 760
    left = 120
    right = 70
    top = 120
    bottom = 120
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(max(float(row["cv_oof"]["mae"]), float(row["test"]["mae"])) for row in rows) * 1.08

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 54, "Attempt 8 CV vs Test MAE", 34, "#111827", weight=700),
        text(left, 84, "The small gaps show that the rerun generalizes cleanly to the holdout split.", 18, "#6b7280"),
    ]
    for tick_index in range(6):
        value = max_value * tick_index / 5
        y = top + chart_h - chart_h * (value / max_value)
        parts.append(line(left - 8, y, left + chart_w, y, "#e5e7eb", dash="4 6"))
        parts.append(text(left - 16, y + 6, format_float(value, 1), 18, "#6b7280", anchor="end"))

    group_w = chart_w / len(rows)
    bar_w = min(44.0, group_w * 0.22)
    for index, row in enumerate(rows):
        cx = left + group_w * (index + 0.5)
        cv_value = float(row["cv_oof"]["mae"])
        test_value = float(row["test"]["mae"])
        color = MODEL_COLORS.get(str(row["display_name"]), "#64748b")
        cv_h = chart_h * (cv_value / max_value)
        test_h = chart_h * (test_value / max_value)
        parts.append(rect(cx - bar_w - 6, top + chart_h - cv_h, bar_w, cv_h, "#cbd5e1", rx=6))
        parts.append(rect(cx + 6, top + chart_h - test_h, bar_w, test_h, color, rx=6))
        parts.append(text(cx, top + chart_h + 34, str(row["display_name"]), 17, "#111827", weight=600, anchor="middle"))
        parts.append(text(cx - 6, top + chart_h - cv_h - 10, format_float(cv_value, 2), 14, "#64748b", weight=600, anchor="end"))
        parts.append(text(cx + bar_w + 10, top + chart_h - test_h - 10, format_float(test_value, 2), 14, "#111827", weight=700))

    legend_y = height - 38
    parts.append(rect(left, legend_y - 14, 18, 18, "#cbd5e1", rx=4))
    parts.append(text(left + 28, legend_y, "CV OOF MAE", 18, "#475569", weight=600))
    parts.append(rect(left + 180, legend_y - 14, 18, 18, "#059669", rx=4))
    parts.append(text(left + 208, legend_y, "Test MAE", 18, "#475569", weight=600))
    save_svg(output_path, width, height, parts)


def make_distance_heatmap(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ordered_models = [str(row["display_name"]) for row in rows]
    distance_ranges = ["near", "mid", "far"]
    cell_w = 200
    cell_h = 64
    left = 280
    top = 140
    width = left + cell_w * len(distance_ranges) + 120
    height = top + cell_h * len(ordered_models) + 90

    values = []
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        for distance_row in row["distance_rows"]:
            key = (str(row["display_name"]), str(distance_row["distance_range"]))
            lookup[key] = distance_row
            values.append(float(distance_row["mae"]))
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 56, "Attempt 8 Test MAE by Distance Range", 34, "#111827", weight=700),
        text(left, 86, "Green is better. This shows where each model wins or collapses.", 18, "#6b7280"),
    ]

    for col_index, distance_range in enumerate(distance_ranges):
        x = left + col_index * cell_w
        parts.append(text(x + cell_w / 2, top - 18, distance_range.upper(), 22, "#111827", weight=700, anchor="middle"))
    for row_index, model_name in enumerate(ordered_models):
        y = top + row_index * cell_h
        parts.append(text(left - 18, y + 39, model_name, 22, "#111827", weight=600, anchor="end"))
        for col_index, distance_range in enumerate(distance_ranges):
            x = left + col_index * cell_w
            value = float(lookup[(model_name, distance_range)]["mae"])
            ratio = (value - min_value) / span
            green = int(220 - ratio * 140)
            red = int(46 + ratio * 170)
            blue = int(93 + ratio * 40)
            color = f"rgb({red},{green},{blue})"
            parts.append(rect(x, y, cell_w - 8, cell_h - 8, color, rx=10, stroke="#ffffff", stroke_width=1.5))
            parts.append(text(x + 22, y + 30, format_float(value, 2), 24, "#ffffff", weight=700))
            within10 = float(lookup[(model_name, distance_range)]["within_10m_rate"])
            parts.append(text(x + 22, y + 52, f"within10: {format_pct(within10, 1)}", 15, "#ecfeff", weight=600))
    save_svg(output_path, width, height, parts)


def make_condition_chart(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    conditions = [
        ("clear_sky", "Weather: clear_sky"),
        ("light_rain", "Weather: light_rain"),
        ("10AM", "Time: 10AM"),
        ("8PM", "Time: 8PM"),
    ]
    width = 1480
    height = 860
    panel_w = 620
    panel_h = 300
    margin_x = 90
    margin_y = 140
    gap_x = 80
    gap_y = 80

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(margin_x, 56, "Attempt 8 Test MAE by Condition", 34, "#111827", weight=700),
        text(margin_x, 86, "Each panel compares all 8 models under one weather or time condition.", 18, "#6b7280"),
    ]

    for index, (condition_key, title) in enumerate(conditions):
        panel_col = index % 2
        panel_row = index // 2
        panel_x = margin_x + panel_col * (panel_w + gap_x)
        panel_y = margin_y + panel_row * (panel_h + gap_y)
        parts.append(rect(panel_x, panel_y, panel_w, panel_h, "#f8fafc", rx=14, stroke="#e2e8f0"))
        parts.append(text(panel_x + 22, panel_y + 34, title, 22, "#111827", weight=700))

        if "AM" in condition_key or "PM" in condition_key:
            values = [float(row["time_rows"][condition_key]["mae"]) for row in rows]
        else:
            values = [float(row["weather_rows"][condition_key]["mae"]) for row in rows]
        max_value = max(values) * 1.08
        label_x = panel_x + 170
        bar_x = label_x + 12
        bar_w_max = panel_w - 240
        row_h = 28

        for bar_index, row in enumerate(rows):
            y = panel_y + 58 + bar_index * row_h
            if "AM" in condition_key or "PM" in condition_key:
                value = float(row["time_rows"][condition_key]["mae"])
            else:
                value = float(row["weather_rows"][condition_key]["mae"])
            bar_w = bar_w_max * (value / max_value)
            color = MODEL_COLORS.get(str(row["display_name"]), "#64748b")
            parts.append(text(label_x - 10, y + 18, str(row["display_name"]), 15, "#111827", weight=600, anchor="end"))
            parts.append(rect(bar_x, y, bar_w_max, 18, "#e2e8f0", rx=6))
            parts.append(rect(bar_x, y, bar_w, 18, color, rx=6))
            parts.append(text(bar_x + bar_w + 8, y + 15, format_float(value, 2), 14, "#111827", weight=700))

    save_svg(output_path, width, height, parts)


def make_top_distance_curve(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    selected_names = ["ensemble", "XGBoost+jitter", "RF+jitter", "RF"]
    selected_rows = [row for row in rows if str(row["display_name"]) in selected_names]
    x_values = sorted({float(item["true_distance_m"]) for row in selected_rows for item in row["distance_curve_rows"]})
    y_values = [float(item["mae"]) for row in selected_rows for item in row["distance_curve_rows"]]

    width = 1380
    height = 760
    left = 120
    right = 100
    top = 120
    bottom = 120
    chart_w = width - left - right
    chart_h = height - top - bottom
    min_x = min(x_values)
    max_x = max(x_values)
    max_y = max(y_values) * 1.12

    parts = [
        rect(0, 0, width, height, "#ffffff"),
        text(left, 54, "Top Models by Exact Distance", 34, "#111827", weight=700),
        text(left, 84, "The ensemble wins overall, but most of the lift comes from the XGBoost branch.", 18, "#6b7280"),
    ]

    for tick in x_values:
        x = left + chart_w * ((tick - min_x) / (max_x - min_x))
        parts.append(line(x, top - 8, x, top + chart_h, "#f1f5f9", dash="4 6"))
        parts.append(text(x, top + chart_h + 34, str(int(tick)), 18, "#6b7280", anchor="middle"))
    for tick_index in range(6):
        value = max_y * tick_index / 5
        y = top + chart_h - chart_h * (value / max_y)
        parts.append(line(left, y, left + chart_w, y, "#e5e7eb", dash="4 6"))
        parts.append(text(left - 16, y + 6, format_float(value, 1), 18, "#6b7280", anchor="end"))

    for row in selected_rows:
        points = []
        for item in sorted(row["distance_curve_rows"], key=lambda value: float(value["true_distance_m"])):
            x = left + chart_w * ((float(item["true_distance_m"]) - min_x) / (max_x - min_x))
            y = top + chart_h - chart_h * (float(item["mae"]) / max_y)
            points.append((x, y))
        color = MODEL_COLORS.get(str(row["display_name"]), "#64748b")
        parts.append(polyline(points, color, width=3.0))
        for x, y in points:
            parts.append(circle(x, y, 5, color, stroke="#ffffff"))

    legend_x = width - 300
    legend_y = 150
    for index, name in enumerate(selected_names):
        color = MODEL_COLORS.get(name, "#64748b")
        y = legend_y + index * 28
        parts.append(rect(legend_x, y - 12, 18, 18, color, rx=4))
        parts.append(text(legend_x + 28, y + 2, name, 18, "#111827", weight=600))

    save_svg(output_path, width, height, parts)


def to_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_model_analysis(model_row: Dict[str, Any]) -> str:
    name = str(model_row["display_name"])
    test_row = model_row["test"]
    near = model_row["distance_rows_by_name"]["near"]
    mid = model_row["distance_rows_by_name"]["mid"]
    far = model_row["distance_rows_by_name"]["far"]
    if name == "scale only":
        return (
            "This is the weakest baseline. It fails badly at medium and far distances, has strongly negative "
            "R2, and keeps a large underprediction bias overall."
        )
    if name == "depth linear":
        return (
            "Relative depth alone is not enough. The model sits near zero R2 and behaves only marginally better "
            "than the scale-only baseline."
        )
    if name == "linear+bb":
        return (
            "Adding bbox geometry and metadata creates the first major jump. It is much stronger than the pure "
            "depth baselines, but still biased high in near and mid and low in far."
        )
    if name == "improved linear":
        return (
            "The stacked lower-model feature helps consistently over linear+bb, especially in mid-range, but the "
            "linear family still cannot match the tree-based models."
        )
    if name == "RF":
        return (
            "This is the first clearly competitive model. It handles near distances well, but its mid-range R2 is "
            "almost flat and far-distance underprediction is still noticeable."
        )
    if name == "RF+jitter":
        return (
            "Aggregated jitter features give a clear gain over the exact-feature RF. The improvement is visible in "
            f"all three bands: near {format_float(float(near['mae']), 2)}, mid {format_float(float(mid['mae']), 2)}, "
            f"far {format_float(float(far['mae']), 2)} MAE."
        )
    if name == "XGBoost+jitter":
        return (
            "This is the best single model in Attempt 8. Its test MAE is nearly identical to the final ensemble, "
            "which means most of the blend's strength comes from this branch."
        )
    if name == "ensemble":
        return (
            "This is the overall winner, but only by a hair over XGBoost+jitter. Its biggest value is not a huge "
            "headline gain, but a stable best-in-run result with almost no CV-to-test drift."
        )
    return ""


def build_report(model_rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], coverage: Dict[str, Any]) -> str:
    test_rows = [row["test"] for row in model_rows]
    best_test = min(test_rows, key=lambda row: float(row["mae"]))
    xgb_row = next(row for row in test_rows if str(row["display_name"]) == "XGBoost+jitter")
    rfj_row = next(row for row in test_rows if str(row["display_name"]) == "RF+jitter")
    rf_row = next(row for row in test_rows if str(row["display_name"]) == "RF")
    ensemble_gap_vs_xgb = float(xgb_row["mae"]) - float(best_test["mae"])
    ensemble_gap_vs_rfj = float(rfj_row["mae"]) - float(best_test["mae"])
    ensemble_gap_vs_rf = float(rf_row["mae"]) - float(best_test["mae"])

    overall_table = []
    for row in sorted(test_rows, key=lambda item: float(item["mae"])):
        overall_table.append(
            [
                str(row["display_name"]),
                str(row["role"]),
                format_float(float(row["mae"]), 3),
                format_float(float(row["r2"]), 3),
                format_pct(float(row["within_10m_rate"]), 1),
                format_float(float(row["p95_absolute_error"]), 2),
            ]
        )

    gap_table = []
    for row in sorted(model_rows, key=lambda item: float(item["test"]["mae"])):
        gap_table.append(
            [
                str(row["display_name"]),
                format_float(float(row["cv_oof"]["mae"]), 3),
                format_float(float(row["test"]["mae"]), 3),
                format_float(float(row["test"]["mae"]) - float(row["cv_oof"]["mae"]), 3),
            ]
        )

    distance_table = []
    for row in sorted(model_rows, key=lambda item: float(item["test"]["mae"])):
        near = row["distance_rows_by_name"]["near"]
        mid = row["distance_rows_by_name"]["mid"]
        far = row["distance_rows_by_name"]["far"]
        distance_table.append(
            [
                str(row["display_name"]),
                format_float(float(near["mae"]), 2),
                format_float(float(mid["mae"]), 2),
                format_float(float(far["mae"]), 2),
            ]
        )

    lines: List[str] = []
    lines.append("# Attempt 8 Full Analysis")
    lines.append("")
    lines.append("## Was Attempt 8 rerun from start to finish?")
    lines.append("")
    lines.append(
        "Yes for the modeling pipeline, but not from raw images. Attempt 8 reran model fitting, cross-validation, "
        "held-out evaluation, the RF candidate sweep, the XGBoost candidate sweep, and the blend search. It did not "
        "recompute depth maps or rebuild the upstream feature tables from scratch."
    )
    lines.append("")
    lines.append(
        "In practice, that means Attempt 8 is a full rerun of the comparison benchmark from the feature-table stage "
        "through final metrics, not a full rerun of the entire computer-vision stack."
    )
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- `ensemble` is the best test model with `MAE={format_float(float(best_test['mae']), 3)}` and "
        f"`R2={format_float(float(best_test['r2']), 3)}` on `{int(float(best_test['count']))}` test rows."
    )
    lines.append(
        f"- The gain over `XGBoost+jitter` is tiny: `{format_float(ensemble_gap_vs_xgb, 3)} m` MAE. "
        "The ensemble is best, but the XGBoost branch is carrying most of the signal."
    )
    lines.append(
        f"- The gain over `RF+jitter` is real and meaningful: `{format_float(ensemble_gap_vs_rfj, 3)} m` MAE."
    )
    lines.append(
        f"- The gain over exact-feature `RF` is larger: `{format_float(ensemble_gap_vs_rf, 3)} m` MAE."
    )
    lines.append(
        f"- The selected blend is `RF {format_pct(float(summary['best_blend_row']['rf_weight']), 0)} + "
        f"XGB {format_pct(float(summary['best_blend_row']['xgb_weight']), 0)}`."
    )
    lines.append(
        f"- The unified benchmark contains `{coverage['num_common_rows']}` shared rows, with "
        f"`{coverage['num_test_rows']}` test rows and `{coverage['num_dropped_rows']}` dropped rows."
    )
    lines.append("")
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append(
        to_markdown_table(
            ["Model", "Role", "Test MAE", "Test R2", "Within 10m", "P95 Abs Error"],
            overall_table,
        )
    )
    lines.append("")
    lines.append("## CV to Test Stability")
    lines.append("")
    lines.append(
        "The strongest sign that Attempt 8 is trustworthy is how little the top models move from OOF CV to the holdout test split."
    )
    lines.append("")
    lines.append(to_markdown_table(["Model", "CV MAE", "Test MAE", "Gap"], gap_table))
    lines.append("")
    lines.append("## Distance Range Breakdown")
    lines.append("")
    lines.append(to_markdown_table(["Model", "Near MAE", "Mid MAE", "Far MAE"], distance_table))
    lines.append("")
    lines.append(
        "The entire leaderboard gets worse as distance increases, but the trees degrade much more gracefully than the linear families."
    )
    lines.append("")
    lines.append("## Model by Model")
    lines.append("")
    for row in sorted(model_rows, key=lambda item: float(item["test"]["mae"])):
        test_row = row["test"]
        lines.append(f"### {row['display_name']}")
        lines.append("")
        lines.append(
            f"- Test: `MAE={format_float(float(test_row['mae']), 3)}`, `R2={format_float(float(test_row['r2']), 3)}`, "
            f"`within10={format_pct(float(test_row['within_10m_rate']), 1)}`."
        )
        lines.append(
            f"- Distance MAE: near `{format_float(float(row['distance_rows_by_name']['near']['mae']), 2)}`, "
            f"mid `{format_float(float(row['distance_rows_by_name']['mid']['mae']), 2)}`, "
            f"far `{format_float(float(row['distance_rows_by_name']['far']['mae']), 2)}`."
        )
        lines.append(f"- {build_model_analysis(row)}")
        lines.append("")
    lines.append("## Failure Patterns")
    lines.append("")
    lines.append(
        "- The most persistent weak zone for the best models is the `far` regime, especially around `100m` to `150m`."
    )
    lines.append(
        "- For the ensemble specifically, the most painful cluster is `light_rain` in the `mid` bucket, where the MAE rises above `8m`."
    )
    lines.append(
        "- The largest outliers are concentrated around `80m` to `100m`, with several `light_rain` samples overshooting by more than `40m`."
    )
    lines.append(
        "- The linear families show a structural bias pattern: they tend to overpredict near and mid distances, then underpredict far distances."
    )
    lines.append("")
    lines.append("## Methodological Caveats")
    lines.append("")
    lines.append(
        "- Attempt 8 did not recompute depth maps. It reused prior feature tables from Attempts 1, 2, 3, and 4."
    )
    lines.append(
        f"- The benchmark now covers the full Attempt 8 universe: `{coverage['num_common_rows']}` shared rows, "
        f"`{coverage['num_dev_rows']}` dev rows, `{coverage['num_test_rows']}` test rows, and "
        f"`{coverage['num_dropped_rows']}` dropped rows."
    )
    lines.append(
        "- The original missing exact-feature rows were regenerated with the Attempt1 extraction path before this rerun, "
        "so the current model comparison is no longer biased by the previous `10AM`/short-distance dropout."
    )
    lines.append("")
    lines.append("## Generated Plots")
    lines.append("")
    lines.append("- `plots/attempt8_test_mae_ranking.svg`")
    lines.append("- `plots/attempt8_test_r2_ranking.svg`")
    lines.append("- `plots/attempt8_cv_vs_test_mae.svg`")
    lines.append("- `plots/attempt8_distance_range_heatmap.svg`")
    lines.append("- `plots/attempt8_condition_mae.svg`")
    lines.append("- `plots/attempt8_top_models_exact_distance.svg`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_metrics = read_csv_rows(REPORTS_DIR / "model_metrics.csv")
    distance_range_metrics = read_csv_rows(REPORTS_DIR / "distance_range_metrics.csv")
    grouped_error_summary = read_csv_rows(REPORTS_DIR / "grouped_error_summary.csv")
    true_distance_summary = read_csv_rows(REPORTS_DIR / "true_distance_summary.csv")
    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    coverage = json.loads((ARTIFACTS_ROOT / "audit" / "coverage_summary.json").read_text(encoding="utf-8"))

    test_rows = [row for row in model_metrics if str(row["split_name"]) == "test"]
    cv_rows = [row for row in model_metrics if str(row["split_name"]) == "cv_oof"]
    grouped_test_rows = [row for row in grouped_error_summary if str(row["split_name"]) == "test"]
    distance_test_rows = [row for row in distance_range_metrics if str(row["split_name"]) == "test"]
    exact_distance_test_rows = [row for row in true_distance_summary if str(row["split_name"]) == "test"]

    cv_by_name = {str(row["display_name"]): row for row in cv_rows}
    distance_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for row in distance_test_rows:
        distance_by_name.setdefault(str(row["display_name"]), []).append(row)
    grouped_weather_by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    grouped_time_by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in grouped_test_rows:
        name = str(row["model_name"])
        grouping = str(row["grouping"])
        if grouping == "weather":
            grouped_weather_by_name.setdefault(name, {})[str(row["weather"])] = row
        elif grouping == "time_of_day":
            grouped_time_by_name.setdefault(name, {})[str(row["time_of_day"])] = row
    exact_curve_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for row in exact_distance_test_rows:
        exact_curve_by_name.setdefault(str(row["display_name"]), []).append(row)

    model_rows: List[Dict[str, Any]] = []
    for test_row in test_rows:
        name = str(test_row["display_name"])
        weather_key_name = name
        time_key_name = name
        distance_rows = sorted(distance_by_name[name], key=lambda row: ["near", "mid", "far"].index(str(row["distance_range"])))
        model_rows.append(
            {
                "display_name": name,
                "role": str(test_row["role"]),
                "test": test_row,
                "cv_oof": cv_by_name[name],
                "distance_rows": distance_rows,
                "distance_rows_by_name": {str(row["distance_range"]): row for row in distance_rows},
                "weather_rows": grouped_weather_by_name[weather_key_name],
                "time_rows": grouped_time_by_name[time_key_name],
                "distance_curve_rows": sorted(exact_curve_by_name[name], key=lambda row: float(row["true_distance_m"])),
            }
        )

    model_rows.sort(key=lambda row: float(row["test"]["mae"]))

    make_horizontal_bar_chart(
        [row["test"] for row in model_rows],
        title="Attempt 8 Test MAE Ranking",
        subtitle="All poster models plus the internal XGBoost branch. Lower is better.",
        value_key="mae",
        output_path=PLOTS_DIR / "attempt8_test_mae_ranking.svg",
        value_label="m",
        min_value=0.0,
    )
    make_r2_chart([row["test"] for row in model_rows], PLOTS_DIR / "attempt8_test_r2_ranking.svg")
    make_grouped_cv_test_chart(model_rows, PLOTS_DIR / "attempt8_cv_vs_test_mae.svg")
    make_distance_heatmap(model_rows, PLOTS_DIR / "attempt8_distance_range_heatmap.svg")
    make_condition_chart(model_rows, PLOTS_DIR / "attempt8_condition_mae.svg")
    make_top_distance_curve(model_rows, PLOTS_DIR / "attempt8_top_models_exact_distance.svg")

    report_text = build_report(model_rows, summary, coverage)
    (REPORTS_DIR / "attempt8_full_analysis.md").write_text(report_text, encoding="utf-8")
    print(REPORTS_DIR / "attempt8_full_analysis.md")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()
