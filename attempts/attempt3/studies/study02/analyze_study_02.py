"""
Generate an analysis package for attempt3 Study 02.
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
UPPER_RF_CANDIDATE_METRICS_PATH = REPORTS_DIR / "upper_rf_candidate_metrics.csv"
UPPER_RF_IMPORTANCES_PATH = REPORTS_DIR / "upper_rf_feature_importances.csv"

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


def _metric_dict(
    absolute_errors: np.ndarray,
    relative_errors: np.ndarray,
    signed_errors: np.ndarray,
) -> Dict[str, float]:
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
        absolute_errors = np.asarray(
            [float(row["absolute_error_m"]) for row in group_rows],
            dtype=np.float64,
        )
        relative_errors = np.asarray(
            [float(row["relative_error"]) for row in group_rows],
            dtype=np.float64,
        )
        signed_errors = np.asarray(
            [float(row["signed_error_m"]) for row in group_rows],
            dtype=np.float64,
        )
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


def _build_overall_rows(
    test_metrics: Dict[str, Any],
    cv_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
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
        model_rows = [
            row for row in exact_distance_rows if str(row["model_name"]) == model_name
        ]
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
    mre_values = [
        100.0 * float(row["test_mean_relative_error"]) for row in overall_rows
    ]
    colors = [
        MODEL_COLORS.get(str(row["model_name"]), "#4e79a7") for row in overall_rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(labels))

    axes[0].bar(x, mae_values, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Test MAE (m)")
    axes[0].set_title("Study 02 Overall Test MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, mre_values, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Test Mean Relative Error (%)")
    axes[1].set_title("Study 02 Overall Test Relative Error")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distance_range_comparison(
    distance_range_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    range_order = ["near", "mid", "far"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(range_order))
    width = 0.22

    for index, model_name in enumerate(FOCUS_MODELS):
        model_rows = [
            row for row in distance_range_rows if str(row["model_name"]) == model_name
        ]
        mae_values = [
            float(
                next(
                    row
                    for row in model_rows
                    if row["distance_range"] == distance_range
                )["mae"]
            )
            for distance_range in range_order
        ]
        mre_values = [
            100.0
            * float(
                next(
                    row
                    for row in model_rows
                    if row["distance_range"] == distance_range
                )["mean_relative_error"]
            )
            for distance_range in range_order
        ]
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            mae_values,
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
        axes[1].bar(
            x + offset,
            mre_values,
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(range_order)
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("Study 02 Distance-Range MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(range_order)
    axes[1].set_ylabel("Mean Relative Error (%)")
    axes[1].set_title("Study 02 Distance-Range Relative Error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_candidate_sweep(
    candidate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x_labels = ["rf_shallow", "rf_medium", "rf_deep", "rf_unbounded"]
    x = np.arange(len(x_labels))

    for model_name in FOCUS_MODELS:
        model_rows = [
            row for row in candidate_rows if str(row["model_name"]) == model_name
        ]
        mae_values = [
            float(
                next(
                    row
                    for row in model_rows
                    if row["candidate_name"] == candidate_name
                )["oof_mae"]
            )
            for candidate_name in x_labels
        ]
        mre_values = [
            100.0
            * float(
                next(
                    row
                    for row in model_rows
                    if row["candidate_name"] == candidate_name
                )["oof_mean_relative_error"]
            )
            for candidate_name in x_labels
        ]
        axes[0].plot(
            x,
            mae_values,
            marker="o",
            linewidth=2.2,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
        axes[1].plot(
            x,
            mre_values,
            marker="o",
            linewidth=2.2,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=15)
    axes[0].set_ylabel("CV OOF MAE (m)")
    axes[0].set_title("Study 02 Upper-RF Candidate Sweep: MAE")
    axes[0].grid(alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=15)
    axes[1].set_ylabel("CV OOF Mean Relative Error (%)")
    axes[1].set_title("Study 02 Upper-RF Candidate Sweep: Relative Error")
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
        row
        for row in importance_rows
        if str(row["model_name"]) == model_name
        and str(row["candidate_name"]) == candidate_name
    ]
    filtered = sorted(
        filtered,
        key=lambda row: float(row["importance_value"]),
        reverse=True,
    )[:14]
    labels = [str(row["term"]) for row in filtered]
    values = [float(row["importance_value"]) for row in filtered]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=MODEL_COLORS.get(model_name, "#4e79a7"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Study 02 Winner Feature Importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_markdown_summary(
    summary: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> str:
    winner = str(summary["winner_model_name"])
    winner_candidate = str(summary["winner_candidate_name"])
    winner_test = test_metrics["models"][winner]["overall"]
    stacked = test_metrics["models"]["stacked_rf_depth_geometry_metadata_rf"]["overall"]
    rawgeom = test_metrics["models"]["raw_depth_geometry_metadata_rf"]["overall"]
    geometry = test_metrics["models"]["geometry_metadata_rf"]["overall"]
    depth_only = test_metrics["models"]["small_rf_depth_only_rf"]["overall"]
    raw_only = test_metrics["models"]["raw_depth_only_rf"]["overall"]
    return "\n".join(
        [
            "# Study 02 Analysis",
            "",
            "## Headline",
            "",
            "- Winner: `{}` with candidate `{}`.".format(winner, winner_candidate),
            "- Winner test MAE: `{:.4f}m`".format(float(winner_test["mae"])),
            "- Winner mean relative error: `{:.2f}%`".format(
                100.0 * float(winner_test["mean_relative_error"])
            ),
            "",
            "## Key Takeaways",
            "",
            "- The best final model was `raw_depth_geometry_metadata_rf`, not the stacked version.",
            "- The stacked model stayed very strong, but it was slightly worse than the raw-depth big RF.",
            "- Geometry remained the dominant signal once the upper model became a Random Forest.",
            "- The main performance jump from Study 01 to Study 02 came from upgrading the top-level model family to Random Forest.",
            "",
            "## Important Numbers",
            "",
            "- Winner test MAE: `{:.4f}m`".format(float(rawgeom["mae"])),
            "- Stacked test MAE: `{:.4f}m`".format(float(stacked["mae"])),
            "- Geometry-only test MAE: `{:.4f}m`".format(float(geometry["mae"])),
            "- Small-RF-depth-only test MAE: `{:.4f}m`".format(float(depth_only["mae"])),
            "- Raw-depth-only test MAE: `{:.4f}m`".format(float(raw_only["mae"])),
            "",
            "## Interpretation",
            "",
            "- The small RF clearly improved the depth-only signal, but that advantage did not translate into the best final stacked RF.",
            "- Once the upper model could already learn strong non-linear corrections from geometry and raw depth together, the extra lower-model depth feature was no longer the winner.",
            "",
        ]
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary = _read_json(SUMMARY_PATH)
    test_metrics = _read_json(TEST_METRICS_PATH)
    cv_metrics = _read_json(CV_METRICS_PATH)
    test_prediction_rows = _read_csv_rows(TEST_PREDICTIONS_PATH)
    candidate_metric_rows = _read_csv_rows(UPPER_RF_CANDIDATE_METRICS_PATH)
    importance_rows = _load_feature_importance_rows(UPPER_RF_IMPORTANCES_PATH)

    exact_distance_rows = _group_predictions_by_distance(test_prediction_rows)
    overall_rows = _build_overall_rows(test_metrics, cv_metrics)
    distance_range_rows = _build_distance_range_rows(test_metrics)
    candidate_rows = _build_candidate_rows(candidate_metric_rows)

    _write_csv_rows(
        exact_distance_rows,
        ANALYSIS_DIR / "study02_exact_distance_test_metrics.csv",
    )
    _write_csv_rows(
        overall_rows,
        ANALYSIS_DIR / "study02_overall_model_metrics.csv",
    )
    _write_csv_rows(
        distance_range_rows,
        ANALYSIS_DIR / "study02_distance_range_metrics.csv",
    )
    _write_csv_rows(
        candidate_rows,
        ANALYSIS_DIR / "study02_candidate_sweep_metrics.csv",
    )

    _plot_metric_by_distance(
        exact_distance_rows,
        metric_key="mae",
        ylabel="MAE (m)",
        title="Study 02 Test MAE by Exact Distance",
        output_path=ANALYSIS_DIR / "study02_exact_distance_mae_all_models.png",
    )
    _plot_metric_by_distance(
        exact_distance_rows,
        metric_key="mean_relative_error",
        ylabel="Mean Relative Error",
        title="Study 02 Test Relative Error by Exact Distance",
        output_path=ANALYSIS_DIR / "study02_exact_distance_relative_error_all_models.png",
    )
    _plot_metric_by_distance(
        exact_distance_rows,
        metric_key="mean_signed_error",
        ylabel="Mean Signed Error (m)",
        title="Study 02 Test Signed Error by Exact Distance",
        output_path=ANALYSIS_DIR / "study02_exact_distance_signed_error_all_models.png",
    )
    _plot_overall_model_comparison(
        overall_rows,
        ANALYSIS_DIR / "study02_overall_model_comparison.png",
    )
    _plot_distance_range_comparison(
        distance_range_rows,
        ANALYSIS_DIR / "study02_distance_range_comparison.png",
    )
    _plot_candidate_sweep(
        candidate_rows,
        ANALYSIS_DIR / "study02_candidate_sweep.png",
    )
    _plot_winner_feature_importance(
        importance_rows,
        model_name=str(summary["winner_model_name"]),
        candidate_name=str(summary["winner_candidate_name"]),
        output_path=ANALYSIS_DIR / "study02_winner_feature_importances.png",
    )

    _write_text(
        ANALYSIS_DIR / "study_02_analysis.md",
        _build_markdown_summary(summary, test_metrics),
    )

    print("[study02-analysis] complete")
    print("  analysis_dir: {}".format(ANALYSIS_DIR))


if __name__ == "__main__":
    main()
