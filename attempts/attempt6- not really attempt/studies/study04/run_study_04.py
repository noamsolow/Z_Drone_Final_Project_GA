from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_ROOT = REPO_ROOT / "attempts" / "attempt6" / "studies" / "study04" / "artifacts"
PLOTS_DIR = OUTPUT_ROOT / "plots"
REPORTS_DIR = OUTPUT_ROOT / "reports"


def _read_csv_rows(input_path: Path) -> List[Dict[str, str]]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(input_path: Path) -> Dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def _write_csv_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _sorted_distance_values(rows: Sequence[Dict[str, Any]]) -> List[float]:
    distance_values = {float(row["true_distance_m"]) for row in rows}
    return sorted(distance_values)


def _aggregate_by_distance(
    rows: Sequence[Dict[str, str]],
    model_label: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[float, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["true_distance_m"])].append(row)

    output_rows: List[Dict[str, Any]] = []
    for distance_value in sorted(grouped.keys()):
        bucket_rows = grouped[distance_value]
        output_rows.append(
            {
                "model_label": model_label,
                "true_distance_m": distance_value,
                "count": len(bucket_rows),
                "mae": _round_float(_mean(float(row["absolute_error_m"]) for row in bucket_rows)),
                "mean_relative_error": _round_float(
                    _mean(float(row["relative_error"]) for row in bucket_rows)
                ),
            }
        )
    return output_rows


def _style_axes(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_overall_progress_plot(
    milestone_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x_values = list(range(len(milestone_rows)))
    y_values = [float(row[metric_key]) for row in milestone_rows]
    labels = [str(row["short_label"]) for row in milestone_rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_values, y_values, color="#1f4e79", linewidth=2.5, marker="o", markersize=7)
    ax.fill_between(x_values, y_values, [0.0] * len(y_values), color="#d8e8f5", alpha=0.35)

    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            "{:.2f}".format(y_value),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model stage")
    ax.set_title("Overall Project Progress by Model Stage", fontsize=13, pad=12)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, hspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_distance_curve_plot(
    curve_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [
        "#7f8c8d",
        "#bdc3c7",
        "#1f4e79",
        "#2a7f62",
        "#c17c00",
        "#8b1e3f",
        "#5b4b8a",
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[str(row["model_label"])].append(row)

    for color_index, model_label in enumerate(grouped.keys()):
        rows = sorted(grouped[model_label], key=lambda row: float(row["true_distance_m"]))
        ax.plot(
            [float(row["true_distance_m"]) for row in rows],
            [float(row[metric_key]) for row in rows],
            label=model_label,
            color=colors[color_index % len(colors)],
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    _style_axes(
        ax=ax,
        title="Synthetic Milestone Models by True Distance",
        ylabel=ylabel,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _attempt4_rf_baseline_metrics() -> Dict[str, float]:
    test_metrics = _read_json(
        REPO_ROOT
        / "attempts"
        / "attempt4"
        / "studies"
        / "study03"
        / "artifacts"
        / "reports"
        / "test_metrics.json"
    )
    overall = test_metrics["models"]["rf_winner_baseline"]["overall"]
    return {
        "mae": float(overall["mae"]),
        "mean_relative_error": float(overall["mean_relative_error"]),
    }


def _calibration_split_row(
    split_eval_summary_csv: Path,
    calibration_model: str,
    calibration_fraction: float,
) -> Dict[str, str]:
    rows = _read_csv_rows(split_eval_summary_csv)
    for row in rows:
        if (
            str(row["calibration_model"]) == calibration_model
            and math.isclose(float(row["calibration_fraction"]), calibration_fraction, rel_tol=0.0, abs_tol=1e-12)
        ):
            return row
    raise ValueError(
        "Calibration row not found for model '{}' fraction '{}'".format(
            calibration_model,
            calibration_fraction,
        )
    )


def _filter_prediction_rows(
    input_path: Path,
    model_name: str,
) -> List[Dict[str, str]]:
    rows = _read_csv_rows(input_path)
    return [row for row in rows if str(row.get("model_name", "")) == model_name]


def build_milestone_rows() -> List[Dict[str, Any]]:
    attempt4_rf_baseline = _attempt4_rf_baseline_metrics()
    calibration_row = _calibration_split_row(
        split_eval_summary_csv=REPO_ROOT
        / "attempts"
        / "attempt5"
        / "studies"
        / "study04"
        / "artifacts"
        / "reports"
        / "split_eval_summary.csv",
        calibration_model="per_drone_quadratic",
        calibration_fraction=0.2,
    )

    return [
        {
            "stage_index": 1,
            "short_label": "Scale only",
            "full_label": "Attempt 1 scale_only",
            "dataset_scope": "synthetic",
            "comparison_scope": "synthetic full-model baseline",
            "mae": 47.74601066154458,
            "mean_relative_error": 0.7511225617955533,
            "source": "attempts/attempt1/artifacts/reports/test_metrics.json",
        },
        {
            "stage_index": 2,
            "short_label": "Depth linear",
            "full_label": "Attempt 1 depth_only_linear",
            "dataset_scope": "synthetic",
            "comparison_scope": "synthetic full-model baseline",
            "mae": 32.111035175478015,
            "mean_relative_error": 0.655736410423863,
            "source": "attempts/attempt1/artifacts/reports/test_metrics.json",
        },
        {
            "stage_index": 3,
            "short_label": "Linear + BB",
            "full_label": "Attempt 1 depth_geometry_metadata_linear",
            "dataset_scope": "synthetic",
            "comparison_scope": "synthetic full-model baseline",
            "mae": 14.50476668725308,
            "mean_relative_error": 0.27793528149447283,
            "source": "attempts/attempt1/artifacts/reports/test_metrics.json",
        },
        {
            "stage_index": 4,
            "short_label": "Improved linear",
            "full_label": "Attempt 3 Study01 stacked_rf_depth_geometry_metadata_linear",
            "dataset_scope": "synthetic",
            "comparison_scope": "synthetic full-model progress",
            "mae": 12.550933180117003,
            "mean_relative_error": 0.1968225342330095,
            "source": "attempts/attempt3/studies/study01/artifacts/reports/summary.json",
        },
        {
            "stage_index": 5,
            "short_label": "RF",
            "full_label": "Attempt 4 Study03 rf_winner_baseline",
            "dataset_scope": "synthetic",
            "comparison_scope": "harder detector-like benchmark",
            "mae": attempt4_rf_baseline["mae"],
            "mean_relative_error": attempt4_rf_baseline["mean_relative_error"],
            "source": "attempts/attempt4/studies/study03/artifacts/reports/test_metrics.json",
        },
        {
            "stage_index": 6,
            "short_label": "Noisy-bbox RF",
            "full_label": "Attempt 3 Study03 raw_depth_geometry_metadata_rf",
            "dataset_scope": "synthetic",
            "comparison_scope": "synthetic full-model progress",
            "mae": 5.210499766037984,
            "mean_relative_error": 0.07434651658151488,
            "source": "attempts/attempt3/studies/study03/artifacts/reports/summary.json",
        },
        {
            "stage_index": 7,
            "short_label": "RF+XGB ensemble",
            "full_label": "Attempt 4 Study03 equal_weight_blend",
            "dataset_scope": "synthetic",
            "comparison_scope": "harder detector-like benchmark",
            "mae": 7.632477833812474,
            "mean_relative_error": 0.09884744302837088,
            "source": "attempts/attempt4/studies/study03/artifacts/reports/test_metrics.json",
        },
        {
            "stage_index": 8,
            "short_label": "Calibrated ensemble",
            "full_label": "Attempt 5 Study04 per_drone_quadratic honest 20% split evaluation",
            "dataset_scope": "real Nenrus",
            "comparison_scope": "real calibrated benchmark",
            "mae": float(calibration_row["mae_mean"]),
            "mean_relative_error": float(calibration_row["mean_relative_error_mean"]),
            "source": "attempts/attempt5/studies/study04/artifacts/reports/split_eval_summary.csv",
        },
    ]


def build_synthetic_curve_rows() -> List[Dict[str, Any]]:
    curve_specs = [
        {
            "model_label": "Scale only",
            "input_csv": REPO_ROOT / "attempts" / "attempt1" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "scale_only",
        },
        {
            "model_label": "Depth linear",
            "input_csv": REPO_ROOT / "attempts" / "attempt1" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "depth_only_linear",
        },
        {
            "model_label": "Linear + BB",
            "input_csv": REPO_ROOT / "attempts" / "attempt1" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "depth_geometry_metadata_linear",
        },
        {
            "model_label": "Improved linear",
            "input_csv": REPO_ROOT / "attempts" / "attempt3" / "studies" / "study01" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "stacked_rf_depth_geometry_metadata_linear",
        },
        {
            "model_label": "RF",
            "input_csv": REPO_ROOT / "attempts" / "attempt4" / "studies" / "study03" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "rf_winner_baseline",
        },
        {
            "model_label": "Noisy-bbox RF",
            "input_csv": REPO_ROOT / "attempts" / "attempt3" / "studies" / "study03" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "raw_depth_geometry_metadata_rf__rf_unbounded",
        },
        {
            "model_label": "RF+XGB ensemble",
            "input_csv": REPO_ROOT / "attempts" / "attempt4" / "studies" / "study03" / "artifacts" / "reports" / "test_predictions.csv",
            "model_name": "equal_weight_blend",
        },
    ]

    curve_rows: List[Dict[str, Any]] = []
    for spec in curve_specs:
        prediction_rows = _filter_prediction_rows(
            input_path=Path(spec["input_csv"]),
            model_name=str(spec["model_name"]),
        )
        if not prediction_rows:
            raise ValueError("No prediction rows found for '{}'".format(spec["model_label"]))
        curve_rows.extend(_aggregate_by_distance(prediction_rows, str(spec["model_label"])))
    return curve_rows


def build_analysis_text(
    milestone_rows: Sequence[Dict[str, Any]],
    synthetic_curve_rows: Sequence[Dict[str, Any]],
) -> str:
    best_overall = min(milestone_rows, key=lambda row: float(row["mae"]))
    best_synthetic = min(
        [row for row in milestone_rows if str(row["dataset_scope"]) == "synthetic"],
        key=lambda row: float(row["mae"]),
    )
    lines = [
        "# Attempt6 Study04 Analysis",
        "",
        "## Overview Graphs",
        "",
        "The overall progress graphs use the eight agreed milestone models.",
        "They should be treated as a narrative project-progress figure rather than one strict apples-to-apples benchmark, because the final two stages switch to a harder detector-like setup and then to real calibrated Nenrus evaluation.",
        "",
        "- Best overall milestone by MAE: `{}` at {:.4f} m".format(
            best_overall["full_label"],
            float(best_overall["mae"]),
        ),
        "- Best synthetic milestone by MAE: `{}` at {:.4f} m".format(
            best_synthetic["full_label"],
            float(best_synthetic["mae"]),
        ),
        "",
        "## Synthetic Distance Curves",
        "",
        "The synthetic distance-curve plots only use synthetic models and therefore support a cleaner distance-based comparison.",
        "They are built directly from saved prediction rows and aggregate MAE and mean relative error at each true distance.",
        "",
        "Included synthetic models:",
    ]
    for model_label in sorted({str(row["model_label"]) for row in synthetic_curve_rows}):
        lines.append("- `{}`".format(model_label))
    lines.append("")
    lines.append("The real calibrated ensemble is intentionally excluded from the synthetic distance curves.")
    return "\n".join(lines) + "\n"


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    milestone_rows = build_milestone_rows()
    synthetic_curve_rows = build_synthetic_curve_rows()

    _write_csv_rows(milestone_rows, REPORTS_DIR / "milestone_overview.csv")
    _write_csv_rows(synthetic_curve_rows, REPORTS_DIR / "synthetic_distance_curves.csv")

    _save_overall_progress_plot(
        milestone_rows=milestone_rows,
        metric_key="mae",
        ylabel="MAE (m)",
        output_path=PLOTS_DIR / "overall_progress_mae.png",
    )
    _save_overall_progress_plot(
        milestone_rows=milestone_rows,
        metric_key="mean_relative_error",
        ylabel="Mean relative error",
        output_path=PLOTS_DIR / "overall_progress_mean_relative_error.png",
    )
    _save_distance_curve_plot(
        curve_rows=synthetic_curve_rows,
        metric_key="mae",
        ylabel="MAE (m)",
        output_path=PLOTS_DIR / "synthetic_distance_mae.png",
    )
    _save_distance_curve_plot(
        curve_rows=synthetic_curve_rows,
        metric_key="mean_relative_error",
        ylabel="Mean relative error",
        output_path=PLOTS_DIR / "synthetic_distance_mean_relative_error.png",
    )

    analysis_text = build_analysis_text(milestone_rows, synthetic_curve_rows)
    (REPORTS_DIR / "analysis.md").write_text(analysis_text, encoding="utf-8")

    summary = {
        "study_name": "study04_milestone_progress_graphs",
        "num_milestone_models": len(milestone_rows),
        "num_synthetic_distance_models": len({row["model_label"] for row in synthetic_curve_rows}),
        "plots": [
            str(PLOTS_DIR / "overall_progress_mae.png"),
            str(PLOTS_DIR / "overall_progress_mean_relative_error.png"),
            str(PLOTS_DIR / "synthetic_distance_mae.png"),
            str(PLOTS_DIR / "synthetic_distance_mean_relative_error.png"),
        ],
        "milestone_overview_csv": str(REPORTS_DIR / "milestone_overview.csv"),
        "synthetic_distance_curves_csv": str(REPORTS_DIR / "synthetic_distance_curves.csv"),
        "analysis_md": str(REPORTS_DIR / "analysis.md"),
    }
    _write_json(summary, REPORTS_DIR / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
