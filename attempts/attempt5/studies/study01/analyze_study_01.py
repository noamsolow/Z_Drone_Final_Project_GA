"""
Generate a graph package for attempt5 Study 01.
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
MODEL_SCHEMA_PATH = REPORTS_DIR / "model_schema.json"
TEST_METRICS_PATH = REPORTS_DIR / "dev_holdout_test_metrics.json"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "dev_holdout_test_predictions.csv"

MODEL_ORDER = ["equal_weight_blend", "tuned_weight_blend"]
MODEL_LABELS = {
    "equal_weight_blend": "Equal-Weight Blend",
    "tuned_weight_blend": "Tuned-Weight Blend",
}
MODEL_COLORS = {
    "equal_weight_blend": "#4e79a7",
    "tuned_weight_blend": "#e15759",
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


def _group_predictions_by_distance(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model_name"]).replace("attempt5_dev_holdout_", ""), float(row["true_distance_m"]))
        grouped[key].append(row)
    output_rows: List[Dict[str, Any]] = []
    for (model_name, distance_m), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
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


def _build_range_rows(test_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        for distance_range in ["near", "mid", "far"]:
            metrics = test_metrics[model_name]["per_distance_range"][distance_range]
            rows.append(
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
    return rows


def _build_jitter_rows(prediction_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        model_name = str(row["model_name"]).replace("attempt5_dev_holdout_", "")
        jitter_name = str(row["image_path"]).split("::jitter=")[-1]
        grouped[(model_name, jitter_name)].append(row)
    output_rows: List[Dict[str, Any]] = []
    for (model_name, jitter_name), group_rows in sorted(grouped.items()):
        absolute_errors = np.asarray([float(row["absolute_error_m"]) for row in group_rows], dtype=np.float64)
        relative_errors = np.asarray([float(row["relative_error"]) for row in group_rows], dtype=np.float64)
        signed_errors = np.asarray([float(row["signed_error_m"]) for row in group_rows], dtype=np.float64)
        output_rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "jitter_name": jitter_name,
                **_metric_dict(absolute_errors, relative_errors, signed_errors),
            }
        )
    return output_rows


def _plot_overall(test_metrics: Dict[str, Any], output_path: Path) -> None:
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    mae = [float(test_metrics[name]["overall"]["mae"]) for name in MODEL_ORDER]
    mre = [100.0 * float(test_metrics[name]["overall"]["mean_relative_error"]) for name in MODEL_ORDER]
    colors = [MODEL_COLORS[name] for name in MODEL_ORDER]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(x, mae, color=colors)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("Test MAE (m)")
    axes[0].set_title("Attempt5 Study01 Test MAE")

    axes[1].bar(x, mre, color=colors)
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Attempt5 Study01 Relative Error")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_metric_by_distance(
    rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = sorted({str(row["model_name"]) for row in rows})
    for model_name in model_names:
        model_rows = [row for row in rows if str(row["model_name"]) == model_name]
        distances = [float(row["true_distance_m"]) for row in model_rows]
        values = [float(row[metric_key]) for row in model_rows]
        if metric_key == "mean_relative_error":
            values = [100.0 * value for value in values]
        ax.plot(distances, values, marker="o", linewidth=2.0, markersize=4, label=MODEL_LABELS.get(model_name, model_name), color=MODEL_COLORS.get(model_name, "#4e79a7"))
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_distance_ranges(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["near", "mid", "far"]
    for idx, model_name in enumerate(MODEL_ORDER):
        model_rows = [row for row in rows if str(row["model_name"]) == model_name]
        values = [next(float(row["mae"]) for row in model_rows if str(row["distance_range"]) == category) for category in categories]
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])
    ax.set_xticks(x, [category.title() for category in categories])
    ax.set_ylabel("Test MAE (m)")
    ax.set_title("Attempt5 Study01 Distance Range Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_jitter_mae(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    jitter_names = sorted({str(row["jitter_name"]) for row in rows})
    x = np.arange(len(jitter_names))
    model_names = sorted({str(row["model_name"]) for row in rows})
    width = 0.8 / max(len(model_names), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model_name in enumerate(model_names):
        model_rows = [row for row in rows if str(row["model_name"]) == model_name]
        values = [next(float(row["mae"]) for row in model_rows if str(row["jitter_name"]) == jitter_name) for jitter_name in jitter_names]
        offset = idx - ((len(model_names) - 1) / 2.0)
        ax.bar(x + offset * width, values, width=width, label=MODEL_LABELS.get(model_name, model_name), color=MODEL_COLORS.get(model_name, "#4e79a7"))
    ax.set_xticks(x, jitter_names, rotation=20, ha="right")
    ax.set_ylabel("Test MAE (m)")
    ax.set_title("Attempt5 Study01 MAE by Jitter Variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_true_vs_predicted(prediction_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    filtered = [row for row in prediction_rows if str(row["model_name"]).endswith("tuned_weight_blend")]
    true_values = np.asarray([float(row["true_distance_m"]) for row in filtered], dtype=np.float64)
    predicted_values = np.asarray([float(row["predicted_distance_m"]) for row in filtered], dtype=np.float64)
    line_min = float(min(true_values.min(), predicted_values.min()))
    line_max = float(max(true_values.max(), predicted_values.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_values, predicted_values, s=10, alpha=0.2, color="#4e79a7")
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black")
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Predicted Distance (m)")
    ax.set_title("Attempt5 Study01 Tuned Blend True vs Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary = _read_json(SUMMARY_PATH)
    model_schema = _read_json(MODEL_SCHEMA_PATH)
    test_metrics = _read_json(TEST_METRICS_PATH)
    prediction_rows = _read_csv_rows(TEST_PREDICTIONS_PATH)

    exact_rows = _group_predictions_by_distance(prediction_rows)
    range_rows = _build_range_rows(test_metrics)
    jitter_rows = _build_jitter_rows(prediction_rows)

    _write_csv_rows(exact_rows, ANALYSIS_DIR / "study01_exact_distance_metrics.csv")
    _write_csv_rows(range_rows, ANALYSIS_DIR / "study01_distance_range_metrics.csv")
    _write_csv_rows(jitter_rows, ANALYSIS_DIR / "study01_jitter_metrics.csv")

    _plot_overall(test_metrics, ANALYSIS_DIR / "study01_overall_blend_comparison.png")
    _plot_metric_by_distance(
        exact_rows,
        "mae",
        "MAE (m)",
        "Attempt5 Study01 Winner MAE by Exact Distance",
        ANALYSIS_DIR / "study01_exact_distance_mae.png",
    )
    _plot_metric_by_distance(
        exact_rows,
        "mean_relative_error",
        "Mean Relative Error (%)",
        "Attempt5 Study01 Winner Relative Error by Exact Distance",
        ANALYSIS_DIR / "study01_exact_distance_relative_error.png",
    )
    _plot_distance_ranges(range_rows, ANALYSIS_DIR / "study01_distance_range_comparison.png")
    _plot_jitter_mae(jitter_rows, ANALYSIS_DIR / "study01_jitter_mae_comparison.png")
    _plot_true_vs_predicted(prediction_rows, ANALYSIS_DIR / "study01_tuned_blend_true_vs_predicted.png")

    analysis_md = "\n".join(
        [
            "# Attempt5 Study01 Analysis",
            "",
            f"- Model name: `{summary['model_name']}`",
            f"- RF candidate: `{summary['rf_candidate_name']}`",
            f"- XGB candidate: `{summary['xgb_candidate_name']}`",
            f"- Tuned blend weights: `{summary['tuned_blend_weights']}`",
            f"- Equal blend weights: `{summary['equal_blend_weights']}`",
            f"- Tabular features: `{summary['num_tabular_features']}`",
            f"- Continuous features: `{len(model_schema['continuous_feature_names'])}`",
            f"- Categorical features: `{len(model_schema['categorical_feature_names'])}`",
        ]
    )
    _write_text(ANALYSIS_DIR / "study_01_analysis.md", analysis_md)


if __name__ == "__main__":
    main()
