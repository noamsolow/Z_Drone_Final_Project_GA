"""
Generate an analysis package for attempt3 Study 01.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPORTS_DIR = STUDY_DIR / "artifacts" / "reports"
ANALYSIS_DIR = REPORTS_DIR / "analysis"

SUMMARY_PATH = REPORTS_DIR / "summary.json"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
CV_METRICS_PATH = REPORTS_DIR / "cv_metrics.json"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
LOWER_MODEL_SUMMARY_PATH = REPORTS_DIR / "lower_model_summary.json"

MODEL_ORDER = [
    "raw_depth_only_linear",
    "geometry_metadata_linear",
    "raw_depth_geometry_metadata_linear",
    "small_rf_depth_only_linear",
    "stacked_rf_depth_geometry_metadata_linear",
]
MODEL_LABELS = {
    "raw_depth_only_linear": "Raw Depth Only",
    "geometry_metadata_linear": "Geometry + Metadata",
    "raw_depth_geometry_metadata_linear": "Raw Depth + Geometry + Metadata",
    "small_rf_depth_only_linear": "Small RF Depth Only",
    "stacked_rf_depth_geometry_metadata_linear": "Stacked RF Depth + Geometry + Metadata",
}
MODEL_COLORS = {
    "raw_depth_only_linear": "#9c755f",
    "geometry_metadata_linear": "#4e79a7",
    "raw_depth_geometry_metadata_linear": "#59a14f",
    "small_rf_depth_only_linear": "#f28e2b",
    "stacked_rf_depth_geometry_metadata_linear": "#e15759",
}
BASELINES_FOR_IMPROVEMENT = [
    "raw_depth_geometry_metadata_linear",
    "geometry_metadata_linear",
    "small_rf_depth_only_linear",
    "raw_depth_only_linear",
]


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv_rows(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    materialized = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _metric_dict(absolute_errors: np.ndarray, relative_errors: np.ndarray, signed_errors: np.ndarray) -> Dict[str, float]:
    if absolute_errors.size == 0:
        return {
            "count": 0,
            "mae": 0.0,
            "mean_relative_error": 0.0,
            "median_relative_error": 0.0,
            "mean_signed_error": 0.0,
            "rmse": 0.0,
        }
    return {
        "count": int(absolute_errors.size),
        "mae": float(np.mean(absolute_errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "median_relative_error": float(np.median(relative_errors)),
        "mean_signed_error": float(np.mean(signed_errors)),
        "rmse": float(np.sqrt(np.mean(signed_errors ** 2))),
    }


def _group_test_predictions_by_distance(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_name"]), float(row["true_distance_m"]))].append(row)

    output_rows: List[Dict[str, Any]] = []
    for (model_name, distance_m), group_rows in sorted(grouped.items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), item[0][1])):
        absolute_errors = np.asarray([float(row["absolute_error_m"]) for row in group_rows], dtype=np.float64)
        relative_errors = np.asarray([float(row["relative_error"]) for row in group_rows], dtype=np.float64)
        signed_errors = np.asarray([float(row["signed_error_m"]) for row in group_rows], dtype=np.float64)
        metrics = _metric_dict(absolute_errors, relative_errors, signed_errors)
        output_rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "true_distance_m": float(distance_m),
                **metrics,
            }
        )
    return output_rows


def _build_overall_metric_rows(test_metrics: Dict[str, Any], cv_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        if model_name not in test_metrics["models"]:
            continue
        test_overall = test_metrics["models"][model_name]["overall"]
        cv_overall = cv_metrics["models"][model_name]["oof_overall"]
        output_rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "test_mae": float(test_overall["mae"]),
                "test_mean_relative_error": float(test_overall["mean_relative_error"]),
                "test_rmse": float(test_overall["rmse"]),
                "test_r2": float(test_overall["r2"]),
                "cv_oof_mae": float(cv_overall["mae"]),
                "cv_oof_mean_relative_error": float(cv_overall["mean_relative_error"]),
                "cv_oof_rmse": float(cv_overall["rmse"]),
                "cv_oof_r2": float(cv_overall["r2"]),
            }
        )
    return output_rows


def _build_distance_range_rows(test_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        if model_name not in test_metrics["models"]:
            continue
        per_range = test_metrics["models"][model_name]["per_distance_range"]
        for distance_range in ["near", "mid", "far"]:
            metrics = per_range[distance_range]
            output_rows.append(
                {
                    "model_name": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "distance_range": distance_range,
                    "mae": float(metrics["mae"]),
                    "mean_relative_error": float(metrics["mean_relative_error"]),
                    "mean_signed_error": float(metrics["mean_signed_error"]),
                    "count": int(metrics["count"]),
                }
            )
    return output_rows


def _build_improvement_rows(
    exact_distance_rows: Sequence[Dict[str, Any]],
    winner_model_name: str,
) -> List[Dict[str, Any]]:
    by_model_distance = {
        (str(row["model_name"]), float(row["true_distance_m"])): row for row in exact_distance_rows
    }
    winner_distances = sorted(
        {
            float(row["true_distance_m"])
            for row in exact_distance_rows
            if str(row["model_name"]) == winner_model_name
        }
    )
    output_rows: List[Dict[str, Any]] = []
    for baseline_name in BASELINES_FOR_IMPROVEMENT:
        for distance_m in winner_distances:
            winner_row = by_model_distance[(winner_model_name, distance_m)]
            baseline_row = by_model_distance[(baseline_name, distance_m)]
            output_rows.append(
                {
                    "winner_model_name": winner_model_name,
                    "baseline_model_name": baseline_name,
                    "baseline_model_label": MODEL_LABELS.get(baseline_name, baseline_name),
                    "true_distance_m": distance_m,
                    "mae_improvement_m": float(baseline_row["mae"]) - float(winner_row["mae"]),
                    "mean_relative_error_improvement": float(baseline_row["mean_relative_error"])
                    - float(winner_row["mean_relative_error"]),
                    "mean_signed_error_delta": float(winner_row["mean_signed_error"])
                    - float(baseline_row["mean_signed_error"]),
                }
            )
    return output_rows


def _plot_metric_by_distance(
    exact_distance_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name in MODEL_ORDER:
        model_rows = [row for row in exact_distance_rows if str(row["model_name"]) == model_name]
        distances = [float(row["true_distance_m"]) for row in model_rows]
        values = [float(row[metric_key]) for row in model_rows]
        ax.plot(
            distances,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name),
        )
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_overall_model_comparison(
    overall_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [row["model_label"] for row in overall_rows]
    mae_values = [float(row["test_mae"]) for row in overall_rows]
    mre_values = [100.0 * float(row["test_mean_relative_error"]) for row in overall_rows]
    colors = [MODEL_COLORS.get(str(row["model_name"]), "#4e79a7") for row in overall_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(labels))

    axes[0].bar(x, mae_values, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Test MAE (m)")
    axes[0].set_title("Overall Test MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, mre_values, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Overall Test Relative Error")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_stacked_improvement_by_distance(
    improvement_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for baseline_name in BASELINES_FOR_IMPROVEMENT:
        baseline_rows = [
            row for row in improvement_rows if str(row["baseline_model_name"]) == baseline_name
        ]
        distances = [float(row["true_distance_m"]) for row in baseline_rows]
        values = [float(row[metric_key]) for row in baseline_rows]
        ax.plot(
            distances,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label="vs {}".format(MODEL_LABELS.get(baseline_name, baseline_name)),
            color=MODEL_COLORS.get(baseline_name),
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distance_range_comparison(
    distance_range_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    selected_models = [
        "geometry_metadata_linear",
        "raw_depth_geometry_metadata_linear",
        "stacked_rf_depth_geometry_metadata_linear",
    ]
    range_order = ["near", "mid", "far"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(range_order))
    width = 0.24

    for idx, model_name in enumerate(selected_models):
        model_rows = {
            str(row["distance_range"]): row
            for row in distance_range_rows
            if str(row["model_name"]) == model_name
        }
        offset = (idx - 1) * width
        mae_values = [float(model_rows[distance_range]["mae"]) for distance_range in range_order]
        mre_values = [
            100.0 * float(model_rows[distance_range]["mean_relative_error"])
            for distance_range in range_order
        ]
        color = MODEL_COLORS.get(model_name)
        label = MODEL_LABELS.get(model_name, model_name)
        axes[0].bar(x + offset, mae_values, width=width, color=color, label=label)
        axes[1].bar(x + offset, mre_values, width=width, color=color, label=label)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(range_order)
    axes[0].set_ylabel("Test MAE (m)")
    axes[0].set_title("Distance-Range MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(range_order)
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Distance-Range Relative Error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_signed_error_by_distance(
    exact_distance_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name in MODEL_ORDER:
        model_rows = [row for row in exact_distance_rows if str(row["model_name"]) == model_name]
        distances = [float(row["true_distance_m"]) for row in model_rows]
        values = [float(row["mean_signed_error"]) for row in model_rows]
        ax.plot(
            distances,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name),
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Mean Signed Error (m)")
    ax.set_title("Bias by Exact Distance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_analysis() -> Dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary = _read_json(SUMMARY_PATH)
    test_metrics = _read_json(TEST_METRICS_PATH)
    cv_metrics = _read_json(CV_METRICS_PATH)
    lower_model_summary = _read_json(LOWER_MODEL_SUMMARY_PATH)
    test_prediction_rows = _read_csv_rows(TEST_PREDICTIONS_PATH)

    exact_distance_rows = _group_test_predictions_by_distance(test_prediction_rows)
    overall_rows = _build_overall_metric_rows(test_metrics, cv_metrics)
    distance_range_rows = _build_distance_range_rows(test_metrics)
    improvement_rows = _build_improvement_rows(
        exact_distance_rows=exact_distance_rows,
        winner_model_name=str(summary["winner_model_name"]),
    )

    _write_csv_rows(exact_distance_rows, ANALYSIS_DIR / "exact_distance_test_metrics.csv")
    _write_csv_rows(overall_rows, ANALYSIS_DIR / "overall_model_metrics.csv")
    _write_csv_rows(distance_range_rows, ANALYSIS_DIR / "distance_range_test_metrics.csv")
    _write_csv_rows(improvement_rows, ANALYSIS_DIR / "stacked_improvement_by_distance.csv")

    _plot_metric_by_distance(
        exact_distance_rows=exact_distance_rows,
        metric_key="mae",
        ylabel="MAE (m)",
        title="All Models: Test MAE by Exact Distance",
        output_path=ANALYSIS_DIR / "exact_distance_mae_all_models.png",
    )
    _plot_metric_by_distance(
        exact_distance_rows=exact_distance_rows,
        metric_key="mean_relative_error",
        ylabel="Mean Relative Error",
        title="All Models: Test Relative Error by Exact Distance",
        output_path=ANALYSIS_DIR / "exact_distance_relative_error_all_models.png",
    )
    _plot_overall_model_comparison(
        overall_rows=overall_rows,
        output_path=ANALYSIS_DIR / "overall_model_comparison.png",
    )
    _plot_stacked_improvement_by_distance(
        improvement_rows=improvement_rows,
        metric_key="mae_improvement_m",
        ylabel="MAE Improvement (baseline - stacked) [m]",
        title="Stacked Model Improvement over Baselines by Distance",
        output_path=ANALYSIS_DIR / "stacked_improvement_mae_by_distance.png",
    )
    _plot_stacked_improvement_by_distance(
        improvement_rows=improvement_rows,
        metric_key="mean_relative_error_improvement",
        ylabel="Relative Error Improvement (baseline - stacked)",
        title="Stacked Model Relative-Error Improvement over Baselines by Distance",
        output_path=ANALYSIS_DIR / "stacked_improvement_relative_error_by_distance.png",
    )
    _plot_distance_range_comparison(
        distance_range_rows=distance_range_rows,
        output_path=ANALYSIS_DIR / "distance_range_comparison.png",
    )
    _plot_signed_error_by_distance(
        exact_distance_rows=exact_distance_rows,
        output_path=ANALYSIS_DIR / "exact_distance_signed_error_all_models.png",
    )

    winner_model_name = str(summary["winner_model_name"])
    winner_overall = next(row for row in overall_rows if str(row["model_name"]) == winner_model_name)
    raw_depth_overall = next(
        row for row in overall_rows if str(row["model_name"]) == "raw_depth_geometry_metadata_linear"
    )
    geometry_overall = next(
        row for row in overall_rows if str(row["model_name"]) == "geometry_metadata_linear"
    )
    small_rf_overall = next(
        row for row in overall_rows if str(row["model_name"]) == "small_rf_depth_only_linear"
    )

    report_lines = [
        "# Study 01 Analysis",
        "",
        "## Scope",
        "",
        "- Shared rows used: `{}`".format(summary["num_merged_rows"]),
        "- Dev rows: `{}`".format(summary["num_dev_rows"]),
        "- Test rows: `{}`".format(summary["num_test_rows"]),
        "- Selected lower model: `{}`".format(summary["selected_lower_model_name"]),
        "- Lower-model depth features: `{}`".format(lower_model_summary["num_selected_lower_features"]),
        "",
        "## Main Result",
        "",
        "- Winner: `{}`".format(MODEL_LABELS.get(winner_model_name, winner_model_name)),
        "- Test MAE: `{:.2f}m`".format(float(winner_overall["test_mae"])),
        "- Test mean relative error: `{:.2%}`".format(float(winner_overall["test_mean_relative_error"])),
        "- CV OOF MAE: `{:.2f}m`".format(float(winner_overall["cv_oof_mae"])),
        "- CV OOF mean relative error: `{:.2%}`".format(float(winner_overall["cv_oof_mean_relative_error"])),
        "",
        "## Improvement",
        "",
        "- Versus `Raw Depth + Geometry + Metadata`: `+{:.2f}m` MAE improvement and `+{:.2%}` relative-error improvement.".format(
            float(raw_depth_overall["test_mae"]) - float(winner_overall["test_mae"]),
            float(raw_depth_overall["test_mean_relative_error"]) - float(winner_overall["test_mean_relative_error"]),
        ),
        "- Versus `Geometry + Metadata`: `+{:.2f}m` MAE improvement and `+{:.2%}` relative-error improvement.".format(
            float(geometry_overall["test_mae"]) - float(winner_overall["test_mae"]),
            float(geometry_overall["test_mean_relative_error"]) - float(winner_overall["test_mean_relative_error"]),
        ),
        "- Versus `Small RF Depth Only`: `+{:.2f}m` MAE improvement and `+{:.2%}` relative-error improvement.".format(
            float(small_rf_overall["test_mae"]) - float(winner_overall["test_mae"]),
            float(small_rf_overall["test_mean_relative_error"]) - float(winner_overall["test_mean_relative_error"]),
        ),
        "",
        "## Files",
        "",
        "- `study_01_analysis.md`",
        "- `exact_distance_test_metrics.csv`",
        "- `overall_model_metrics.csv`",
        "- `distance_range_test_metrics.csv`",
        "- `stacked_improvement_by_distance.csv`",
        "- `exact_distance_mae_all_models.png`",
        "- `exact_distance_relative_error_all_models.png`",
        "- `overall_model_comparison.png`",
        "- `stacked_improvement_mae_by_distance.png`",
        "- `stacked_improvement_relative_error_by_distance.png`",
        "- `distance_range_comparison.png`",
        "- `exact_distance_signed_error_all_models.png`",
    ]
    _write_text(ANALYSIS_DIR / "study_01_analysis.md", "\n".join(report_lines) + "\n")

    return {
        "analysis_dir": str(ANALYSIS_DIR),
        "winner_model_name": winner_model_name,
        "winner_test_mae": float(winner_overall["test_mae"]),
        "winner_test_mean_relative_error": float(winner_overall["test_mean_relative_error"]),
    }


def main() -> None:
    result = run_analysis()
    print("Study 01 analysis package written to {}".format(result["analysis_dir"]))


if __name__ == "__main__":
    main()
