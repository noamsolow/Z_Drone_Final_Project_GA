"""
Generate an analysis package for attempt3 Study 03, including comparisons to
attempt3 Study 02.
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


STUDY03_DIR = Path(__file__).resolve().parent
STUDY02_DIR = STUDY03_DIR.parent / "study02"

REPORTS03_DIR = STUDY03_DIR / "artifacts" / "reports"
REPORTS02_DIR = STUDY02_DIR / "artifacts" / "reports"
ANALYSIS_DIR = REPORTS03_DIR / "analysis"

SUMMARY03_PATH = REPORTS03_DIR / "summary.json"
TEST_METRICS03_PATH = REPORTS03_DIR / "test_metrics.json"
CV_METRICS03_PATH = REPORTS03_DIR / "cv_metrics.json"
TEST_PREDICTIONS03_PATH = REPORTS03_DIR / "test_predictions.csv"
UPPER_RF_CANDIDATE_METRICS03_PATH = REPORTS03_DIR / "upper_rf_candidate_metrics.csv"
UPPER_RF_IMPORTANCES03_PATH = REPORTS03_DIR / "upper_rf_feature_importances.csv"

SUMMARY02_PATH = REPORTS02_DIR / "summary.json"
TEST_METRICS02_PATH = REPORTS02_DIR / "test_metrics.json"
CV_METRICS02_PATH = REPORTS02_DIR / "cv_metrics.json"
TEST_PREDICTIONS02_PATH = REPORTS02_DIR / "test_predictions.csv"
UPPER_RF_IMPORTANCES02_PATH = REPORTS02_DIR / "upper_rf_feature_importances.csv"

MODEL_ORDER = [
    "raw_depth_only_rf",
    "geometry_metadata_rf",
    "raw_depth_geometry_metadata_rf",
    "small_rf_depth_only_rf",
    "stacked_rf_depth_geometry_metadata_rf",
]
MODEL_LABELS = {
    "raw_depth_only_rf": "Raw Depth Only RF",
    "geometry_metadata_rf": "Geometry + Metadata RF",
    "raw_depth_geometry_metadata_rf": "Raw Depth + Geometry + Metadata RF",
    "small_rf_depth_only_rf": "Small RF Depth Only RF",
    "stacked_rf_depth_geometry_metadata_rf": "Stacked RF Depth + Geometry + Metadata RF",
}
MODEL_COLORS = {
    "raw_depth_only_rf": "#9c755f",
    "geometry_metadata_rf": "#4e79a7",
    "raw_depth_geometry_metadata_rf": "#59a14f",
    "small_rf_depth_only_rf": "#f28e2b",
    "stacked_rf_depth_geometry_metadata_rf": "#e15759",
}
STUDY_COLORS = {
    "study02": "#4e79a7",
    "study03": "#e15759",
}
FOCUS_MODELS = [
    "geometry_metadata_rf",
    "raw_depth_geometry_metadata_rf",
    "stacked_rf_depth_geometry_metadata_rf",
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


def _write_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


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


def _strip_candidate_name(model_name: str) -> str:
    parts = str(model_name).split("__")
    return "__".join(parts[:-1]) if len(parts) > 1 else str(model_name)


def _group_predictions_by_distance(rows: Sequence[Dict[str, Any]], *, strip_candidate: bool) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_name = _strip_candidate_name(str(row["model_name"])) if strip_candidate else str(row["model_name"])
        grouped[(model_name, float(row["true_distance_m"]))].append(row)
    output_rows: List[Dict[str, Any]] = []
    for (model_name, distance_m), group_rows in sorted(
        grouped.items(),
        key=lambda item: (MODEL_ORDER.index(item[0][0]), item[0][1]),
    ):
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


def _build_overall_rows(test_metrics: Dict[str, Any], cv_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _build_group_rows(
    prediction_rows: Sequence[Dict[str, Any]],
    *,
    strip_candidate: bool,
    group_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        model_name = _strip_candidate_name(str(row["model_name"])) if strip_candidate else str(row["model_name"])
        key = [model_name]
        for group_key in group_keys:
            key.append(str(row[group_key]))
        grouped[tuple(key)].append(row)

    output_rows: List[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        model_name = key[0]
        absolute_errors = np.asarray([float(row["absolute_error_m"]) for row in group_rows], dtype=np.float64)
        relative_errors = np.asarray([float(row["relative_error"]) for row in group_rows], dtype=np.float64)
        signed_errors = np.asarray([float(row["signed_error_m"]) for row in group_rows], dtype=np.float64)
        metrics = _metric_dict(absolute_errors, relative_errors, signed_errors)
        out_row: Dict[str, Any] = {
            "model_name": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
        }
        for index, group_key in enumerate(group_keys, start=1):
            out_row[group_key] = key[index]
        out_row.update(metrics)
        output_rows.append(out_row)
    return output_rows


def _build_candidate_rows(candidate_metrics_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row in candidate_metrics_rows:
        output_rows.append(
            {
                "model_name": str(row["model_name"]),
                "model_label": MODEL_LABELS.get(str(row["model_name"]), str(row["model_name"])),
                "candidate_name": str(row["candidate_name"]),
                "oof_mae": float(row["oof_mae"]),
                "oof_mean_relative_error": float(row["oof_mean_relative_error"]),
                "max_depth": str(row.get("param_max_depth", "")),
                "min_samples_leaf": int(float(row.get("param_min_samples_leaf", 0))),
            }
        )
    return output_rows


def _load_feature_importance_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(path)
    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        output_rows.append(
            {
                "model_name": str(row["model_name"]),
                "candidate_name": str(row["candidate_name"]),
                "term": str(row["term"]),
                "importance_value": float(row["importance_value"]),
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


def _plot_overall_model_comparison(overall_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
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
    axes[0].set_title("Study 03 Overall Test MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, mre_values, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Study 03 Overall Test Relative Error")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distance_range_comparison(distance_range_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    range_order = ["near", "mid", "far"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(range_order))
    width = 0.22

    for index, model_name in enumerate(FOCUS_MODELS):
        model_rows = [row for row in distance_range_rows if str(row["model_name"]) == model_name]
        mae_values = [
            float(next(row for row in model_rows if row["distance_range"] == distance_range)["mae"])
            for distance_range in range_order
        ]
        mre_values = [
            100.0
            * float(
                next(row for row in model_rows if row["distance_range"] == distance_range)["mean_relative_error"]
            )
            for distance_range in range_order
        ]
        offset = (index - 1) * width
        axes[0].bar(x + offset, mae_values, width=width, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])
        axes[1].bar(x + offset, mre_values, width=width, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(range_order)
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("Study 03 Distance-Range MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(range_order)
    axes[1].set_ylabel("Mean Relative Error (%)")
    axes[1].set_title("Study 03 Distance-Range Relative Error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_candidate_sweep(candidate_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x_labels = ["rf_shallow", "rf_medium", "rf_deep", "rf_unbounded"]
    x = np.arange(len(x_labels))

    for model_name in FOCUS_MODELS:
        model_rows = [row for row in candidate_rows if str(row["model_name"]) == model_name]
        mae_values = [
            float(next(row for row in model_rows if row["candidate_name"] == candidate_name)["oof_mae"])
            for candidate_name in x_labels
        ]
        mre_values = [
            100.0
            * float(
                next(row for row in model_rows if row["candidate_name"] == candidate_name)["oof_mean_relative_error"]
            )
            for candidate_name in x_labels
        ]
        axes[0].plot(x, mae_values, marker="o", linewidth=2.2, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])
        axes[1].plot(x, mre_values, marker="o", linewidth=2.2, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=15)
    axes[0].set_ylabel("CV OOF MAE (m)")
    axes[0].set_title("Study 03 Upper-RF Candidate Sweep: MAE")
    axes[0].grid(alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=15)
    axes[1].set_ylabel("CV OOF Mean Relative Error (%)")
    axes[1].set_title("Study 03 Upper-RF Candidate Sweep: Relative Error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_winner_feature_importance(
    importance_rows: Sequence[Dict[str, Any]],
    model_name: str,
    candidate_name: str,
    output_path: Path,
) -> None:
    filtered = [
        row for row in importance_rows if str(row["model_name"]) == model_name and str(row["candidate_name"]) == candidate_name
    ]
    filtered = sorted(filtered, key=lambda row: float(row["importance_value"]), reverse=True)[:14]
    labels = [str(row["term"]) for row in filtered]
    values = [float(row["importance_value"]) for row in filtered]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=MODEL_COLORS.get(model_name, "#4e79a7"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Study 03 Winner Feature Importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_weather_time_comparison(
    grouped_rows: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    model_names: Sequence[str],
) -> None:
    labels = sorted({str(row[group_key]) for row in grouped_rows})
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.8))

    for index, model_name in enumerate(model_names):
        model_rows = [row for row in grouped_rows if str(row["model_name"]) == model_name]
        values = [
            float(next(row for row in model_rows if str(row[group_key]) == label)[metric_key])
            for label in labels
        ]
        offset = (index - (len(model_names) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_weather_time_heatmap(
    grouped_rows: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    metric_key: str,
    title: str,
    output_path: Path,
) -> None:
    model_rows = [row for row in grouped_rows if str(row["model_name"]) == model_name]
    weathers = sorted({str(row["weather"]) for row in model_rows})
    times = sorted({str(row["time_of_day"]) for row in model_rows})
    matrix = np.zeros((len(weathers), len(times)), dtype=np.float64)

    for weather_index, weather in enumerate(weathers):
        for time_index, time_of_day in enumerate(times):
            row = next(
                row for row in model_rows
                if str(row["weather"]) == weather and str(row["time_of_day"]) == time_of_day
            )
            value = float(row[metric_key])
            if metric_key == "mean_relative_error":
                value *= 100.0
            matrix[weather_index, time_index] = value

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    image = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels(times)
    ax.set_yticks(np.arange(len(weathers)))
    ax.set_yticklabels(weathers)
    ax.set_title(title)

    for weather_index in range(len(weathers)):
        for time_index in range(len(times)):
            ax.text(
                time_index,
                weather_index,
                "{:.2f}".format(matrix[weather_index, time_index]),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_true_vs_predicted_scatter(
    prediction_rows: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    title: str,
    output_path: Path,
) -> None:
    model_rows = [row for row in prediction_rows if _strip_candidate_name(str(row["model_name"])) == model_name]
    true_values = np.asarray([float(row["true_distance_m"]) for row in model_rows], dtype=np.float64)
    pred_values = np.asarray([float(row["predicted_distance_m"]) for row in model_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(true_values, pred_values, s=18, alpha=0.35, color=MODEL_COLORS.get(model_name))
    lower = float(min(true_values.min(), pred_values.min()))
    upper = float(max(true_values.max(), pred_values.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Predicted Distance (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_error_histogram(
    prediction_rows: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    output_path: Path,
) -> None:
    model_rows = [row for row in prediction_rows if _strip_candidate_name(str(row["model_name"])) == model_name]
    signed_errors = np.asarray([float(row["signed_error_m"]) for row in model_rows], dtype=np.float64)
    absolute_errors = np.asarray([float(row["absolute_error_m"]) for row in model_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].hist(signed_errors, bins=32, color=MODEL_COLORS.get(model_name), alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title("Winner Signed Error Distribution")
    axes[0].set_xlabel("Signed Error (m)")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist(absolute_errors, bins=32, color=MODEL_COLORS.get(model_name), alpha=0.85)
    axes[1].set_title("Winner Absolute Error Distribution")
    axes[1].set_xlabel("Absolute Error (m)")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_feature_group_rows(
    importance_rows: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    candidate_name: str,
) -> List[Dict[str, Any]]:
    rows = [
        row for row in importance_rows
        if str(row["model_name"]) == model_name and str(row["candidate_name"]) == candidate_name
    ]
    grouped: Dict[str, float] = defaultdict(float)
    for row in rows:
        term = str(row["term"])
        importance_value = float(row["importance_value"])
        if term.startswith("weather=") or term.startswith("time_of_day="):
            group_name = "metadata"
        elif "__jitter_std" in term and term.startswith("bbox_"):
            group_name = "geometry_jitter_std"
        elif "__jitter_median" in term and term.startswith("bbox_"):
            group_name = "geometry_jitter_median"
        elif "__jitter_std" in term:
            group_name = "depth_jitter_std"
        elif "__jitter_median" in term:
            group_name = "depth_jitter_median"
        elif term == "depth_lower_model_prediction":
            group_name = "stacked_depth_signal"
        else:
            group_name = "other"
        grouped[group_name] += importance_value

    output_rows = [
        {"feature_group": key, "importance_value": value}
        for key, value in grouped.items()
    ]
    return sorted(output_rows, key=lambda row: float(row["importance_value"]), reverse=True)


def _build_three_bucket_importance_rows(
    importance_rows: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    candidate_name: str,
) -> List[Dict[str, Any]]:
    rows = [
        row for row in importance_rows
        if str(row["model_name"]) == model_name and str(row["candidate_name"]) == candidate_name
    ]
    grouped = {
        "jittered_geometry": 0.0,
        "relative_depth": 0.0,
        "metadata": 0.0,
    }

    for row in rows:
        term = str(row["term"])
        importance_value = float(row["importance_value"])
        if term.startswith("weather=") or term.startswith("time_of_day="):
            grouped["metadata"] += importance_value
        elif term.startswith("bbox_"):
            grouped["jittered_geometry"] += importance_value
        else:
            grouped["relative_depth"] += importance_value

    output_rows = [
        {"feature_group": key, "importance_value": value}
        for key, value in grouped.items()
    ]
    return sorted(output_rows, key=lambda row: float(row["importance_value"]), reverse=True)


def _plot_feature_group_importance(
    feature_group_rows: Sequence[Dict[str, Any]],
    *,
    title: str,
    output_path: Path,
) -> None:
    labels = [str(row["feature_group"]) for row in feature_group_rows]
    values = [float(row["importance_value"]) for row in feature_group_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Total Feature Importance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_three_bucket_importance(
    feature_group_rows: Sequence[Dict[str, Any]],
    *,
    title: str,
    output_path: Path,
) -> None:
    label_map = {
        "jittered_geometry": "Jittered Geometry",
        "relative_depth": "Relative Depth",
        "metadata": "Metadata",
    }
    color_map = {
        "jittered_geometry": "#4e79a7",
        "relative_depth": "#59a14f",
        "metadata": "#f28e2b",
    }
    labels = [label_map.get(str(row["feature_group"]), str(row["feature_group"])) for row in feature_group_rows]
    values = [float(row["importance_value"]) for row in feature_group_rows]
    colors = [color_map.get(str(row["feature_group"]), "#4e79a7") for row in feature_group_rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Feature Importance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.01,
            "{:.3f}".format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_study02_vs_study03_overall(
    summary02: Dict[str, Any],
    test_metrics02: Dict[str, Any],
    summary03: Dict[str, Any],
    test_metrics03: Dict[str, Any],
    output_path: Path,
) -> None:
    winner02 = str(summary02["winner_model_name"])
    winner03 = str(summary03["winner_model_name"])
    rows = [
        ("Study 02 Winner", float(test_metrics02["models"][winner02]["overall"]["mae"]), 100.0 * float(test_metrics02["models"][winner02]["overall"]["mean_relative_error"])),
        ("Study 03 Winner", float(test_metrics03["models"][winner03]["overall"]["mae"]), 100.0 * float(test_metrics03["models"][winner03]["overall"]["mean_relative_error"])),
    ]
    labels = [row[0] for row in rows]
    mae_values = [row[1] for row in rows]
    mre_values = [row[2] for row in rows]
    colors = [STUDY_COLORS["study02"], STUDY_COLORS["study03"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    x = np.arange(len(labels))

    axes[0].bar(x, mae_values, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Test MAE (m)")
    axes[0].set_title("Study 02 vs Study 03 Winner MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, mre_values, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Study 02 vs Study 03 Winner Relative Error")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_study02_vs_study03_exact_distance(
    exact_rows02: Sequence[Dict[str, Any]],
    exact_rows03: Sequence[Dict[str, Any]],
    model_name: str,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for study_name, rows, color in [
        ("Study 02", exact_rows02, STUDY_COLORS["study02"]),
        ("Study 03", exact_rows03, STUDY_COLORS["study03"]),
    ]:
        model_rows = [row for row in rows if str(row["model_name"]) == model_name]
        distances = [float(row["true_distance_m"]) for row in model_rows]
        values = [float(row[metric_key]) for row in model_rows]
        ax.plot(
            distances,
            values,
            marker="o",
            linewidth=2.4,
            markersize=5,
            label=study_name,
            color=color,
        )
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_model_family_shift(
    overall_rows02: Sequence[Dict[str, Any]],
    overall_rows03: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [MODEL_LABELS[name] for name in FOCUS_MODELS]
    mae02 = [float(next(row for row in overall_rows02 if row["model_name"] == name)["test_mae"]) for name in FOCUS_MODELS]
    mae03 = [float(next(row for row in overall_rows03 if row["model_name"] == name)["test_mae"]) for name in FOCUS_MODELS]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2.0, mae02, width=width, label="Study 02", color=STUDY_COLORS["study02"])
    ax.bar(x + width / 2.0, mae03, width=width, label="Study 03", color=STUDY_COLORS["study03"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test MAE (m)")
    ax.set_title("Matched Model Families: Study 02 vs Study 03")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_study02_vs_study03_feature_importance(
    importance_rows02: Sequence[Dict[str, Any]],
    summary02: Dict[str, Any],
    importance_rows03: Sequence[Dict[str, Any]],
    summary03: Dict[str, Any],
    output_path: Path,
) -> None:
    winner02 = str(summary02["winner_model_name"])
    candidate02 = str(summary02["winner_candidate_name"])
    winner03 = str(summary03["winner_model_name"])
    candidate03 = str(summary03["winner_candidate_name"])

    rows02 = [row for row in importance_rows02 if row["model_name"] == winner02 and row["candidate_name"] == candidate02]
    rows03 = [row for row in importance_rows03 if row["model_name"] == winner03 and row["candidate_name"] == candidate03]
    top02 = sorted(rows02, key=lambda row: float(row["importance_value"]), reverse=True)[:10]
    top03 = sorted(rows03, key=lambda row: float(row["importance_value"]), reverse=True)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, rows, title, color in [
        (axes[0], top02, "Study 02 Winner Importances", STUDY_COLORS["study02"]),
        (axes[1], top03, "Study 03 Winner Importances", STUDY_COLORS["study03"]),
    ]:
        labels = [str(row["term"]) for row in rows]
        values = [float(row["importance_value"]) for row in rows]
        y = np.arange(len(labels))
        ax.barh(y, values, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_markdown_summary(
    summary03: Dict[str, Any],
    summary02: Dict[str, Any],
    test_metrics03: Dict[str, Any],
    test_metrics02: Dict[str, Any],
) -> str:
    winner03 = str(summary03["winner_model_name"])
    winner02 = str(summary02["winner_model_name"])
    winner03_test = test_metrics03["models"][winner03]["overall"]
    winner02_test = test_metrics02["models"][winner02]["overall"]
    stacked03 = test_metrics03["models"]["stacked_rf_depth_geometry_metadata_rf"]["overall"]
    stacked02 = test_metrics02["models"]["stacked_rf_depth_geometry_metadata_rf"]["overall"]
    geometry03 = test_metrics03["models"]["geometry_metadata_rf"]["overall"]
    rawgeom03 = test_metrics03["models"]["raw_depth_geometry_metadata_rf"]["overall"]
    return "\n".join(
        [
            "# Study 03 Analysis",
            "",
            "## Headline",
            "",
            "- Study 03 winner: `{}` with test MAE `{:.4f}m` and mean relative error `{:.2f}%`.".format(
                winner03,
                float(winner03_test["mae"]),
                100.0 * float(winner03_test["mean_relative_error"]),
            ),
            "- Study 02 winner: `{}` with test MAE `{:.4f}m` and mean relative error `{:.2f}%`.".format(
                winner02,
                float(winner02_test["mae"]),
                100.0 * float(winner02_test["mean_relative_error"]),
            ),
            "- Study 03 still favored `raw_depth_geometry_metadata_rf`, not the stacked model.",
            "",
            "## Key Takeaways",
            "",
            "- Noisy bbox aggregation improved overall accuracy substantially versus Study 02, despite making the boxes more realistic.",
            "- The stacked model remained very strong, but it still did not beat `raw_depth_geometry_metadata_rf`.",
            "- Geometry remained the dominant signal even after replacing exact bbox measurements with jitter-median and jitter-std features.",
            "- The new `__jitter_std` geometry features became meaningfully important, indicating that robustness/stability information helped the model.",
            "",
            "## Important Numbers",
            "",
            "- Study 03 winner test MAE: `{:.4f}m`".format(float(winner03_test["mae"])),
            "- Study 03 stacked test MAE: `{:.4f}m`".format(float(stacked03["mae"])),
            "- Study 03 geometry-only test MAE: `{:.4f}m`".format(float(geometry03["mae"])),
            "- Gap between Study 03 winner and stacked: `{:.4f}m`".format(
                float(stacked03["mae"]) - float(rawgeom03["mae"])
            ),
            "- Study 02 stacked test MAE: `{:.4f}m`".format(float(stacked02["mae"])),
            "",
            "## Caveat",
            "",
            "- Study 02 and Study 03 are not a perfect apples-to-apples comparison because Study 02 used the smaller shared intersection while Study 03 used all `15,064` images.",
            "- Even so, the comparison is still very informative about how the noisy aggregated feature design changed the overall behavior.",
        ]
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary03 = _read_json(SUMMARY03_PATH)
    test_metrics03 = _read_json(TEST_METRICS03_PATH)
    cv_metrics03 = _read_json(CV_METRICS03_PATH)
    summary02 = _read_json(SUMMARY02_PATH)
    test_metrics02 = _read_json(TEST_METRICS02_PATH)
    cv_metrics02 = _read_json(CV_METRICS02_PATH)

    test_predictions03 = _read_csv_rows(TEST_PREDICTIONS03_PATH)
    test_predictions02 = _read_csv_rows(TEST_PREDICTIONS02_PATH)
    candidate_rows03 = _build_candidate_rows(_read_csv_rows(UPPER_RF_CANDIDATE_METRICS03_PATH))
    importance_rows03 = _load_feature_importance_rows(UPPER_RF_IMPORTANCES03_PATH)
    importance_rows02 = _load_feature_importance_rows(UPPER_RF_IMPORTANCES02_PATH)

    exact_distance_rows03 = _group_predictions_by_distance(test_predictions03, strip_candidate=True)
    exact_distance_rows02 = _group_predictions_by_distance(test_predictions02, strip_candidate=True)
    overall_rows03 = _build_overall_rows(test_metrics03, cv_metrics03)
    overall_rows02 = _build_overall_rows(test_metrics02, cv_metrics02)
    distance_range_rows03 = _build_distance_range_rows(test_metrics03)
    weather_rows03 = _build_group_rows(
        test_predictions03,
        strip_candidate=True,
        group_keys=["weather"],
    )
    time_rows03 = _build_group_rows(
        test_predictions03,
        strip_candidate=True,
        group_keys=["time_of_day"],
    )
    weather_time_rows03 = _build_group_rows(
        test_predictions03,
        strip_candidate=True,
        group_keys=["weather", "time_of_day"],
    )
    winner_feature_group_rows03 = _build_feature_group_rows(
        importance_rows03,
        model_name=str(summary03["winner_model_name"]),
        candidate_name=str(summary03["winner_candidate_name"]),
    )
    winner_three_bucket_rows03 = _build_three_bucket_importance_rows(
        importance_rows03,
        model_name=str(summary03["winner_model_name"]),
        candidate_name=str(summary03["winner_candidate_name"]),
    )

    _write_csv_rows(exact_distance_rows03, ANALYSIS_DIR / "study03_exact_distance_test_metrics.csv")
    _write_csv_rows(exact_distance_rows02, ANALYSIS_DIR / "study02_exact_distance_test_metrics.csv")
    _write_csv_rows(overall_rows03, ANALYSIS_DIR / "study03_overall_model_metrics.csv")
    _write_csv_rows(overall_rows02, ANALYSIS_DIR / "study02_overall_model_metrics.csv")
    _write_csv_rows(distance_range_rows03, ANALYSIS_DIR / "study03_distance_range_metrics.csv")
    _write_csv_rows(candidate_rows03, ANALYSIS_DIR / "study03_candidate_sweep_metrics.csv")
    _write_csv_rows(weather_rows03, ANALYSIS_DIR / "study03_weather_metrics.csv")
    _write_csv_rows(time_rows03, ANALYSIS_DIR / "study03_time_metrics.csv")
    _write_csv_rows(weather_time_rows03, ANALYSIS_DIR / "study03_weather_time_metrics.csv")
    _write_csv_rows(winner_feature_group_rows03, ANALYSIS_DIR / "study03_winner_feature_group_importances.csv")
    _write_csv_rows(winner_three_bucket_rows03, ANALYSIS_DIR / "study03_winner_three_bucket_importances.csv")

    _plot_metric_by_distance(
        exact_distance_rows03,
        metric_key="mae",
        ylabel="MAE (m)",
        title="Study 03: Test MAE by Exact Distance",
        output_path=ANALYSIS_DIR / "study03_exact_distance_mae_all_models.png",
    )
    _plot_metric_by_distance(
        exact_distance_rows03,
        metric_key="mean_relative_error",
        ylabel="Mean Relative Error",
        title="Study 03: Test Relative Error by Exact Distance",
        output_path=ANALYSIS_DIR / "study03_exact_distance_relative_error_all_models.png",
    )
    _plot_metric_by_distance(
        exact_distance_rows03,
        metric_key="mean_signed_error",
        ylabel="Mean Signed Error (m)",
        title="Study 03: Signed Error by Exact Distance",
        output_path=ANALYSIS_DIR / "study03_exact_distance_signed_error_all_models.png",
    )
    _plot_overall_model_comparison(
        overall_rows03,
        output_path=ANALYSIS_DIR / "study03_overall_model_comparison.png",
    )
    _plot_distance_range_comparison(
        distance_range_rows03,
        output_path=ANALYSIS_DIR / "study03_distance_range_comparison.png",
    )
    _plot_candidate_sweep(
        candidate_rows03,
        output_path=ANALYSIS_DIR / "study03_candidate_sweep.png",
    )
    _plot_winner_feature_importance(
        importance_rows03,
        model_name=str(summary03["winner_model_name"]),
        candidate_name=str(summary03["winner_candidate_name"]),
        output_path=ANALYSIS_DIR / "study03_winner_feature_importances.png",
    )
    _plot_weather_time_comparison(
        weather_rows03,
        group_key="weather",
        metric_key="mae",
        ylabel="MAE (m)",
        title="Study 03: Weather Comparison",
        output_path=ANALYSIS_DIR / "study03_weather_comparison.png",
        model_names=FOCUS_MODELS,
    )
    _plot_weather_time_comparison(
        time_rows03,
        group_key="time_of_day",
        metric_key="mean_relative_error",
        ylabel="Mean Relative Error (%)",
        title="Study 03: Time-of-Day Relative Error Comparison",
        output_path=ANALYSIS_DIR / "study03_time_relative_error_comparison.png",
        model_names=FOCUS_MODELS,
    )
    _plot_weather_time_heatmap(
        weather_time_rows03,
        model_name=str(summary03["winner_model_name"]),
        metric_key="mae",
        title="Study 03 Winner MAE by Weather and Time",
        output_path=ANALYSIS_DIR / "study03_winner_weather_time_heatmap_mae.png",
    )
    _plot_weather_time_heatmap(
        weather_time_rows03,
        model_name=str(summary03["winner_model_name"]),
        metric_key="mean_relative_error",
        title="Study 03 Winner Relative Error (%) by Weather and Time",
        output_path=ANALYSIS_DIR / "study03_winner_weather_time_heatmap_relative_error.png",
    )
    _plot_true_vs_predicted_scatter(
        test_predictions03,
        model_name=str(summary03["winner_model_name"]),
        title="Study 03 Winner: True vs Predicted Distance",
        output_path=ANALYSIS_DIR / "study03_winner_true_vs_predicted.png",
    )
    _plot_error_histogram(
        test_predictions03,
        model_name=str(summary03["winner_model_name"]),
        output_path=ANALYSIS_DIR / "study03_winner_error_histograms.png",
    )
    _plot_feature_group_importance(
        winner_feature_group_rows03,
        title="Study 03 Winner Feature-Group Importances",
        output_path=ANALYSIS_DIR / "study03_winner_feature_group_importances.png",
    )
    _plot_three_bucket_importance(
        winner_three_bucket_rows03,
        title="Study 03 Winner: Jittered Geometry vs Relative Depth vs Metadata",
        output_path=ANALYSIS_DIR / "study03_winner_three_bucket_importances.png",
    )
    _plot_study02_vs_study03_overall(
        summary02,
        test_metrics02,
        summary03,
        test_metrics03,
        output_path=ANALYSIS_DIR / "study02_vs_study03_winner_overall.png",
    )
    _plot_model_family_shift(
        overall_rows02,
        overall_rows03,
        output_path=ANALYSIS_DIR / "study02_vs_study03_matched_model_families.png",
    )
    _plot_study02_vs_study03_exact_distance(
        exact_rows02=exact_distance_rows02,
        exact_rows03=exact_distance_rows03,
        model_name="raw_depth_geometry_metadata_rf",
        metric_key="mae",
        ylabel="MAE (m)",
        title="Raw Depth + Geometry + Metadata RF: Study 02 vs Study 03",
        output_path=ANALYSIS_DIR / "study02_vs_study03_rawgeom_mae_by_distance.png",
    )
    _plot_study02_vs_study03_exact_distance(
        exact_rows02=exact_distance_rows02,
        exact_rows03=exact_distance_rows03,
        model_name="stacked_rf_depth_geometry_metadata_rf",
        metric_key="mae",
        ylabel="MAE (m)",
        title="Stacked RF + Geometry + Metadata RF: Study 02 vs Study 03",
        output_path=ANALYSIS_DIR / "study02_vs_study03_stacked_mae_by_distance.png",
    )
    _plot_study02_vs_study03_exact_distance(
        exact_rows02=exact_distance_rows02,
        exact_rows03=exact_distance_rows03,
        model_name="raw_depth_geometry_metadata_rf",
        metric_key="mean_relative_error",
        ylabel="Mean Relative Error",
        title="Raw Depth + Geometry + Metadata RF Relative Error: Study 02 vs Study 03",
        output_path=ANALYSIS_DIR / "study02_vs_study03_rawgeom_relative_error_by_distance.png",
    )
    _plot_study02_vs_study03_feature_importance(
        importance_rows02,
        summary02,
        importance_rows03,
        summary03,
        output_path=ANALYSIS_DIR / "study02_vs_study03_winner_feature_importances.png",
    )

    markdown = _build_markdown_summary(summary03, summary02, test_metrics03, test_metrics02)
    _write_text(ANALYSIS_DIR / "study_03_analysis.md", markdown)

    print("[study03-analysis] wrote {}".format(ANALYSIS_DIR))


if __name__ == "__main__":
    main()
