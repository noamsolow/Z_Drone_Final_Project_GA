"""
Generate a graph package for attempt4 Study 02.
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
UPPER_CANDIDATE_METRICS_PATH = REPORTS_DIR / "upper_xgb_candidate_metrics.csv"
UPPER_IMPORTANCES_PATH = REPORTS_DIR / "upper_xgb_feature_importances.csv"

MODEL_ORDER = [
    "raw_depth_only_xgb",
    "geometry_metadata_xgb",
    "raw_depth_geometry_metadata_xgb",
    "small_xgb_depth_only_xgb",
    "stacked_xgb_depth_geometry_metadata_xgb",
]
MODEL_LABELS = {
    "raw_depth_only_xgb": "Raw Depth Only XGB",
    "geometry_metadata_xgb": "Geometry + Metadata XGB",
    "raw_depth_geometry_metadata_xgb": "Raw Depth + Geometry + Metadata XGB",
    "small_xgb_depth_only_xgb": "Small RF Depth Only XGB",
    "stacked_xgb_depth_geometry_metadata_xgb": "Stacked RF Depth + Geometry + Metadata XGB",
}
MODEL_COLORS = {
    "raw_depth_only_xgb": "#9c755f",
    "geometry_metadata_xgb": "#4e79a7",
    "raw_depth_geometry_metadata_xgb": "#59a14f",
    "small_xgb_depth_only_xgb": "#f28e2b",
    "stacked_xgb_depth_geometry_metadata_xgb": "#e15759",
}


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
        writer.writerows(materialized)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _metric_dict(
    absolute_errors: np.ndarray,
    relative_errors: np.ndarray,
    signed_errors: np.ndarray,
) -> Dict[str, float]:
    return {
        "count": int(absolute_errors.size),
        "mae": float(np.mean(absolute_errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "mean_signed_error": float(np.mean(signed_errors)),
        "rmse": float(np.sqrt(np.mean(signed_errors ** 2))),
    }


def _strip_candidate_name(model_name: str) -> str:
    parts = model_name.split("__")
    return "__".join(parts[:-1]) if len(parts) > 1 else model_name


def _group_predictions_by_distance(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_name = _strip_candidate_name(str(row["model_name"]))
        grouped[(model_name, float(row["true_distance_m"]))].append(row)

    output_rows: List[Dict[str, Any]] = []
    for (model_name, distance_m), group_rows in sorted(
        grouped.items(),
        key=lambda item: (MODEL_ORDER.index(item[0][0]), item[0][1]),
    ):
        absolute_errors = np.asarray([float(row["absolute_error_m"]) for row in group_rows], dtype=np.float64)
        relative_errors = np.asarray([float(row["relative_error"]) for row in group_rows], dtype=np.float64)
        signed_errors = np.asarray([float(row["signed_error_m"]) for row in group_rows], dtype=np.float64)
        output_rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "true_distance_m": float(distance_m),
                **_metric_dict(absolute_errors, relative_errors, signed_errors),
            }
        )
    return output_rows


def _build_overall_rows(test_metrics: Dict[str, Any], cv_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        test_overall = test_metrics["models"][model_name]["overall"]
        cv_overall = cv_metrics["models"][model_name]["oof_overall"]
        output_rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
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
        for distance_range in ["near", "mid", "far"]:
            metrics = test_metrics["models"][model_name]["per_distance_range"][distance_range]
            output_rows.append(
                {
                    "model_name": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "distance_range": distance_range,
                    "mae": float(metrics["mae"]),
                    "mean_relative_error": float(metrics["mean_relative_error"]),
                    "mean_signed_error": float(metrics["mean_signed_error"]),
                    "count": int(metrics["count"]),
                }
            )
    return output_rows


def _load_candidate_rows() -> List[Dict[str, Any]]:
    rows = _read_csv_rows(UPPER_CANDIDATE_METRICS_PATH)
    return [
        {
            "model_name": str(row["model_name"]),
            "model_label": MODEL_LABELS[str(row["model_name"])],
            "candidate_name": str(row["candidate_name"]),
            "oof_mae": float(row["oof_mae"]),
            "oof_mean_relative_error": float(row["oof_mean_relative_error"]),
            "max_depth": str(row.get("param_max_depth", "")),
            "learning_rate": float(row.get("param_learning_rate", 0.0)),
        }
        for row in rows
    ]


def _load_importance_rows(winner_model_name: str, winner_candidate_name: str) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(UPPER_IMPORTANCES_PATH)
    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        if str(row["model_name"]) != winner_model_name or str(row["candidate_name"]) != winner_candidate_name:
            continue
        output_rows.append(
            {
                "term": str(row["term"]),
                "importance_value": float(row["importance_value"]),
            }
        )
    output_rows.sort(key=lambda row: row["importance_value"], reverse=True)
    return output_rows


def _plot_overall(overall_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    labels = [str(row["model_label"]) for row in overall_rows]
    test_mae = [float(row["test_mae"]) for row in overall_rows]
    test_mre = [100.0 * float(row["test_mean_relative_error"]) for row in overall_rows]
    colors = [MODEL_COLORS[str(row["model_name"])] for row in overall_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(x, test_mae, color=colors)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Test MAE (m)")
    axes[0].set_title("Attempt4 Study02 Test MAE")

    axes[1].bar(x, test_mre, color=colors)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Attempt4 Study02 Relative Error")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_metric_by_distance(
    exact_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name in MODEL_ORDER:
        model_rows = [row for row in exact_rows if str(row["model_name"]) == model_name]
        distances = [float(row["true_distance_m"]) for row in model_rows]
        values = [float(row[metric_key]) for row in model_rows]
        if metric_key == "mean_relative_error":
            values = [100.0 * value for value in values]
        ax.plot(distances, values, marker="o", linewidth=2.0, markersize=4, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_distance_ranges(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    categories = ["near", "mid", "far"]
    x = np.arange(len(categories))
    width = 0.16
    for idx, model_name in enumerate(MODEL_ORDER):
        model_rows = [row for row in rows if str(row["model_name"]) == model_name]
        values = [next(float(row["mae"]) for row in model_rows if str(row["distance_range"]) == category) for category in categories]
        ax.bar(x + (idx - 2) * width, values, width=width, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])
    ax.set_xticks(x, [category.title() for category in categories])
    ax.set_ylabel("Test MAE (m)")
    ax.set_title("Attempt4 Study02 Distance Range Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_candidate_sweep(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(rows))
    labels = [f"{row['model_label']}\n{row['candidate_name']}" for row in rows]
    values = [float(row["oof_mae"]) for row in rows]
    colors = [MODEL_COLORS[str(row["model_name"])] for row in rows]
    ax.bar(x, values, color=colors)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("CV OOF MAE (m)")
    ax.set_title("Attempt4 Study02 Candidate Sweep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_importances(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    top_rows = list(rows[:12])
    labels = [str(row["term"]) for row in top_rows][::-1]
    values = [float(row["importance_value"]) for row in top_rows][::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, values, color="#4e79a7")
    ax.set_xlabel("Importance")
    ax.set_title("Attempt4 Study02 Winner Feature Importances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_true_vs_predicted(
    prediction_rows: Sequence[Dict[str, Any]],
    winner_model_name: str,
    output_path: Path,
) -> None:
    filtered = [row for row in prediction_rows if _strip_candidate_name(str(row["model_name"])) == winner_model_name]
    true_values = np.asarray([float(row["true_distance_m"]) for row in filtered], dtype=np.float64)
    predicted_values = np.asarray([float(row["predicted_distance_m"]) for row in filtered], dtype=np.float64)
    line_min = float(min(true_values.min(), predicted_values.min()))
    line_max = float(max(true_values.max(), predicted_values.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_values, predicted_values, s=10, alpha=0.2, color="#4e79a7")
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black")
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Predicted Distance (m)")
    ax.set_title("Attempt4 Study02 Winner True vs Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary = _read_json(SUMMARY_PATH)
    test_metrics = _read_json(TEST_METRICS_PATH)
    cv_metrics = _read_json(CV_METRICS_PATH)
    prediction_rows = _read_csv_rows(TEST_PREDICTIONS_PATH)

    exact_rows = _group_predictions_by_distance(prediction_rows)
    overall_rows = _build_overall_rows(test_metrics, cv_metrics)
    distance_range_rows = _build_distance_range_rows(test_metrics)
    candidate_rows = _load_candidate_rows()
    importance_rows = _load_importance_rows(str(summary["winner_model_name"]), str(summary["winner_candidate_name"]))

    _write_csv_rows(overall_rows, ANALYSIS_DIR / "study02_overall_model_metrics.csv")
    _write_csv_rows(exact_rows, ANALYSIS_DIR / "study02_exact_distance_test_metrics.csv")
    _write_csv_rows(distance_range_rows, ANALYSIS_DIR / "study02_distance_range_metrics.csv")
    _write_csv_rows(candidate_rows, ANALYSIS_DIR / "study02_candidate_sweep_metrics.csv")
    _write_csv_rows(importance_rows, ANALYSIS_DIR / "study02_winner_feature_importances.csv")

    _plot_overall(overall_rows, ANALYSIS_DIR / "study02_overall_model_comparison.png")
    _plot_metric_by_distance(
        exact_rows,
        "mae",
        "MAE (m)",
        "Attempt4 Study02 MAE by Exact Distance",
        ANALYSIS_DIR / "study02_exact_distance_mae_all_models.png",
    )
    _plot_metric_by_distance(
        exact_rows,
        "mean_relative_error",
        "Mean Relative Error (%)",
        "Attempt4 Study02 Relative Error by Exact Distance",
        ANALYSIS_DIR / "study02_exact_distance_relative_error_all_models.png",
    )
    _plot_distance_ranges(distance_range_rows, ANALYSIS_DIR / "study02_distance_range_comparison.png")
    _plot_candidate_sweep(candidate_rows, ANALYSIS_DIR / "study02_candidate_sweep.png")
    _plot_importances(importance_rows, ANALYSIS_DIR / "study02_winner_feature_importances.png")
    _plot_true_vs_predicted(prediction_rows, str(summary["winner_model_name"]), ANALYSIS_DIR / "study02_winner_true_vs_predicted.png")

    winner_test_mae = float(summary["winner_test_mae"])
    winner_test_mre = 100.0 * float(summary["winner_test_mean_relative_error"])
    analysis_md = "\n".join(
        [
            "# Attempt4 Study02 Analysis",
            "",
            f"- Winner model: `{summary['winner_model_name']}`",
            f"- Winner candidate: `{summary['winner_candidate_name']}`",
            f"- Test MAE: `{winner_test_mae:.3f} m`",
            f"- Test mean relative error: `{winner_test_mre:.2f}%`",
            f"- Unique source images: `{summary['num_unique_source_images']}`",
            f"- Rows: `{summary['num_rows']}`",
        ]
    )
    _write_text(ANALYSIS_DIR / "study_02_analysis.md", analysis_md)


if __name__ == "__main__":
    main()
