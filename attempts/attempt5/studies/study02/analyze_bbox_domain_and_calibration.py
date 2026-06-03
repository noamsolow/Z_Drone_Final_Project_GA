"""
Analyze bbox-geometry domain shift and simple Nenrus calibration.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[3]
REPORTS_DIR = STUDY_DIR / "artifacts" / "reports"
PLOTS_DIR = STUDY_DIR / "artifacts" / "plots"

ORIGINAL_FEATURE_TABLE = REPO_ROOT / "attempts" / "attempt4" / "studies" / "study01" / "artifacts" / "features" / "noisy_expanded_feature_table.csv"
NENRUS_FEATURE_TABLE = STUDY_DIR / "artifacts" / "features" / "nenrus_feature_table.csv"
NENRUS_PREDICTIONS = REPORTS_DIR / "predictions.csv"


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _as_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def _quantiles(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _load_geometry_rows() -> List[Dict[str, Any]]:
    original_rows = [
        row for row in _read_csv(ORIGINAL_FEATURE_TABLE)
        if str(row.get("jitter_name", "")) == "original"
    ]
    real_rows = _read_csv(NENRUS_FEATURE_TABLE)

    output: List[Dict[str, Any]] = []
    for row in original_rows:
        output.append(
            {
                "dataset": "Original",
                "dataset_group": "Original",
                "distance_m": float(row["true_distance_m"]),
                "bbox_width_norm": float(row["bbox_width_norm"]),
                "bbox_area_ratio": float(row["bbox_area_ratio"]),
                "bbox_height_norm": float(row["bbox_height_norm"]),
            }
        )
    for row in real_rows:
        drone_type = str(row["drone_type"])
        output.append(
            {
                "dataset": drone_type,
                "dataset_group": "Real",
                "distance_m": float(row["true_distance_m"]),
                "bbox_width_norm": float(row["bbox_width_norm"]),
                "bbox_area_ratio": float(row["bbox_area_ratio"]),
                "bbox_height_norm": float(row["bbox_height_norm"]),
            }
        )
    return output


def _distribution_rows(rows: Sequence[Dict[str, Any]], value_key: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), float(row["distance_m"]))].append(float(row[value_key]))

    output: List[Dict[str, Any]] = []
    for (dataset, distance), values in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        metric_row = {
            "dataset": dataset,
            "distance_m": distance,
            "feature": value_key,
        }
        metric_row.update(_quantiles(values))
        output.append(metric_row)
    return output


def _real_vs_original_rows(distribution_rows: Sequence[Dict[str, Any]], feature: str) -> List[Dict[str, Any]]:
    original_by_distance = {
        float(row["distance_m"]): row
        for row in distribution_rows
        if str(row["dataset"]) == "Original"
    }
    output: List[Dict[str, Any]] = []
    for row in distribution_rows:
        dataset = str(row["dataset"])
        if dataset == "Original":
            continue
        distance = float(row["distance_m"])
        if distance not in original_by_distance:
            continue
        original = original_by_distance[distance]
        real_mean = float(row["mean"])
        original_mean = float(original["mean"])
        ratio = real_mean / original_mean if original_mean != 0.0 else float("nan")
        output.append(
            {
                "dataset": dataset,
                "distance_m": distance,
                "feature": feature,
                "real_count": int(row["count"]),
                "original_count": int(original["count"]),
                "real_mean": real_mean,
                "original_mean": original_mean,
                "real_minus_original_mean": real_mean - original_mean,
                "real_div_original_mean": ratio,
                "real_median": float(row["median"]),
                "original_median": float(original["median"]),
            }
        )
    return sorted(output, key=lambda item: (item["dataset"], item["distance_m"]))


def _plot_width_original_vs_real(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "Original": "#111111",
        "Kongsberg": "#b85c45",
        "Vestfold": "#315f8c",
    }
    markers = {
        "Original": "o",
        "Kongsberg": "s",
        "Vestfold": "^",
    }
    for dataset in ("Original", "Kongsberg", "Vestfold"):
        subset = [row for row in rows if str(row["dataset"]) == dataset]
        ax.scatter(
            [float(row["distance_m"]) for row in subset],
            [float(row["bbox_width_norm"]) for row in subset],
            s=18 if dataset == "Original" else 34,
            alpha=0.18 if dataset == "Original" else 0.75,
            color=colors[dataset],
            marker=markers[dataset],
            label=f"{dataset} samples",
        )

    for dataset in ("Original", "Kongsberg", "Vestfold"):
        grouped: Dict[float, List[float]] = defaultdict(list)
        for row in rows:
            if str(row["dataset"]) == dataset:
                grouped[float(row["distance_m"])].append(float(row["bbox_width_norm"]))
        distances = sorted(grouped)
        means = [float(np.mean(grouped[distance])) for distance in distances]
        ax.plot(
            distances,
            means,
            color=colors[dataset],
            linewidth=2.2,
            label=f"{dataset} mean",
        )

    ax.set_title("Original vs Real: bbox_width_norm by Distance")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("bbox_width_norm")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_area_original_vs_real(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "Original": "#111111",
        "Kongsberg": "#b85c45",
        "Vestfold": "#315f8c",
    }
    for dataset in ("Original", "Kongsberg", "Vestfold"):
        grouped: Dict[float, List[float]] = defaultdict(list)
        for row in rows:
            if str(row["dataset"]) == dataset:
                grouped[float(row["distance_m"])].append(float(row["bbox_area_ratio"]))
        distances = sorted(grouped)
        means = [float(np.mean(grouped[distance])) for distance in distances]
        ax.plot(
            distances,
            means,
            marker="o",
            color=colors[dataset],
            linewidth=2.2,
            label=dataset,
        )
    ax.set_title("Original vs Real: Mean bbox_area_ratio by Distance")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Mean bbox_area_ratio")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_real_div_original(comparison_rows: Sequence[Dict[str, Any]], feature: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for dataset, color in (("Kongsberg", "#b85c45"), ("Vestfold", "#315f8c")):
        subset = [row for row in comparison_rows if str(row["dataset"]) == dataset]
        ax.plot(
            [float(row["distance_m"]) for row in subset],
            [float(row["real_div_original_mean"]) for row in subset],
            marker="o",
            linewidth=2.0,
            label=dataset,
            color=color,
        )
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title(f"Real / Original Mean Ratio: {feature}")
    ax.set_xlabel("Shared distance (m)")
    ax.set_ylabel("real mean / original mean")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _stratified_calibration_split(rows: Sequence[Dict[str, Any]], calibration_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["drone_type"]), float(row["true_distance_m"]))].append(dict(row))

    calibration_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for group_rows in groups.values():
        shuffled = list(group_rows)
        rng.shuffle(shuffled)
        calibration_count = max(1, int(round(len(shuffled) * calibration_fraction)))
        calibration_count = min(calibration_count, len(shuffled) - 1)
        calibration_rows.extend(shuffled[:calibration_count])
        test_rows.extend(shuffled[calibration_count:])
    return calibration_rows, test_rows


def _fit_affine(predictions: Sequence[float], true_values: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(true_values, dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def _fit_scale(predictions: Sequence[float], true_values: Sequence[float]) -> float:
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(true_values, dtype=np.float64)
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return 1.0
    return float(np.dot(x, y) / denominator)


def _evaluate_prediction_rows(rows: Sequence[Dict[str, Any]], predictions: Sequence[float]) -> Dict[str, Any]:
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    errors = preds - true_values
    abs_errors = np.abs(errors)
    rel_errors = np.divide(abs_errors, true_values, out=np.zeros_like(abs_errors), where=true_values > 0)
    return {
        "mae": float(np.mean(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error": float(np.mean(errors)),
        "within_5m_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10m_rate": float(np.mean(abs_errors <= 10.0)),
        "within_20m_rate": float(np.mean(abs_errors <= 20.0)),
    }


def _run_calibration_experiment(rows: Sequence[Dict[str, Any]], calibration_fraction: float = 0.2, num_seeds: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    for seed in range(num_seeds):
        calibration_rows, test_rows = _stratified_calibration_split(rows, calibration_fraction, seed)

        calib_pred = [float(row["predicted_distance_m"]) for row in calibration_rows]
        calib_true = [float(row["true_distance_m"]) for row in calibration_rows]
        test_pred = np.asarray([float(row["predicted_distance_m"]) for row in test_rows], dtype=np.float64)
        test_true_pred = test_pred

        intercept, slope = _fit_affine(calib_pred, calib_true)
        scale = _fit_scale(calib_pred, calib_true)

        per_drone_params: Dict[str, Tuple[float, float]] = {}
        global_params = (intercept, slope)
        for drone_type in sorted({str(row["drone_type"]) for row in calibration_rows}):
            subset = [row for row in calibration_rows if str(row["drone_type"]) == drone_type]
            if len(subset) >= 2:
                per_drone_params[drone_type] = _fit_affine(
                    [float(row["predicted_distance_m"]) for row in subset],
                    [float(row["true_distance_m"]) for row in subset],
                )
            else:
                per_drone_params[drone_type] = global_params

        model_predictions = {
            "raw_uncalibrated": test_true_pred,
            "global_affine": intercept + slope * test_pred,
            "global_scale_only": scale * test_pred,
            "per_drone_affine": np.asarray(
                [
                    per_drone_params[str(row["drone_type"])][0]
                    + per_drone_params[str(row["drone_type"])][1] * float(row["predicted_distance_m"])
                    for row in test_rows
                ],
                dtype=np.float64,
            ),
        }
        for model_name, predictions in model_predictions.items():
            metric_row = {
                "seed": seed,
                "calibration_fraction": calibration_fraction,
                "calibration_count": len(calibration_rows),
                "test_count": len(test_rows),
                "model": model_name,
                "global_intercept": intercept,
                "global_slope": slope,
                "global_scale": scale,
            }
            metric_row.update(_evaluate_prediction_rows(test_rows, predictions))
            run_rows.append(metric_row)

    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["model"])].append(row)
    metric_names = [
        "mae",
        "median_absolute_error",
        "mean_relative_error",
        "median_relative_error",
        "rmse",
        "mean_signed_error",
        "within_5m_rate",
        "within_10m_rate",
        "within_20m_rate",
    ]
    for model_name, model_rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "model": model_name,
            "num_runs": len(model_rows),
            "mean_calibration_count": float(np.mean([float(row["calibration_count"]) for row in model_rows])),
            "mean_test_count": float(np.mean([float(row["test_count"]) for row in model_rows])),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in model_rows], dtype=np.float64)
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
            summary[f"{metric_name}_median"] = float(np.median(values))
        summary_rows.append(summary)
    return run_rows, summary_rows


def _plot_calibration_summary(summary_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(summary_rows, key=lambda row: float(row["mae_mean"]))
    labels = [str(row["model"]) for row in ordered]
    mae_values = [float(row["mae_mean"]) for row in ordered]
    rel_values = [float(row["mean_relative_error_mean"]) for row in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, mae_values, color="#365f46")
    axes[0].set_title("Calibration Experiment: Mean Test MAE")
    axes[0].set_ylabel("MAE (m)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, rel_values, color="#4c6b8a")
    axes[1].set_title("Calibration Experiment: Mean Relative Error")
    axes[1].set_ylabel("Mean relative error")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def run() -> Dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    geometry_rows = _load_geometry_rows()
    _write_csv(REPORTS_DIR / "bbox_geometry_domain_rows.csv", geometry_rows)

    width_distribution = _distribution_rows(geometry_rows, "bbox_width_norm")
    area_distribution = _distribution_rows(geometry_rows, "bbox_area_ratio")
    width_comparison = _real_vs_original_rows(width_distribution, "bbox_width_norm")
    area_comparison = _real_vs_original_rows(area_distribution, "bbox_area_ratio")

    _write_csv(REPORTS_DIR / "bbox_width_norm_distribution_by_dataset_distance.csv", width_distribution)
    _write_csv(REPORTS_DIR / "bbox_area_ratio_distribution_by_dataset_distance.csv", area_distribution)
    _write_csv(REPORTS_DIR / "bbox_width_norm_real_vs_original_shared_distances.csv", width_comparison)
    _write_csv(REPORTS_DIR / "bbox_area_ratio_real_vs_original_shared_distances.csv", area_comparison)

    _plot_width_original_vs_real(geometry_rows, PLOTS_DIR / "bbox_width_norm_original_vs_real_by_distance.png")
    _plot_area_original_vs_real(geometry_rows, PLOTS_DIR / "bbox_area_ratio_original_vs_real_by_distance.png")
    _plot_real_div_original(width_comparison, "bbox_width_norm", PLOTS_DIR / "bbox_width_norm_real_div_original_shared_distances.png")
    _plot_real_div_original(area_comparison, "bbox_area_ratio", PLOTS_DIR / "bbox_area_ratio_real_div_original_shared_distances.png")

    prediction_rows = _read_csv(NENRUS_PREDICTIONS)
    calibration_runs, calibration_summary = _run_calibration_experiment(
        prediction_rows,
        calibration_fraction=0.2,
        num_seeds=100,
    )
    _write_csv(REPORTS_DIR / "simple_calibration_runs.csv", calibration_runs)
    _write_csv(REPORTS_DIR / "simple_calibration_summary.csv", calibration_summary)
    _plot_calibration_summary(calibration_summary, PLOTS_DIR / "simple_calibration_test_performance.png")

    analysis_lines = [
        "# BBox Domain Shift and Simple Calibration Analysis",
        "",
        "## Geometry Comparison",
        "",
        "The original dataset uses only the `jitter_name=original` rows from Attempt 4, so the comparison is between real labelled boxes rather than augmented scale jitter rows.",
        "",
        "Shared distances between Original and Nenrus are `20, 30, 40, 50, 60, 70`.",
        "",
        "### bbox_width_norm: Real / Original Mean Ratio",
        "",
    ]
    for row in width_comparison:
        analysis_lines.append(
            "- `{}` `{}`m: real/original ratio `{}`, real mean `{}`, original mean `{}`".format(
                row["dataset"],
                int(float(row["distance_m"])),
                _fmt(row["real_div_original_mean"]),
                _fmt(row["real_mean"]),
                _fmt(row["original_mean"]),
            )
        )
    analysis_lines.extend(["", "### bbox_area_ratio: Real / Original Mean Ratio", ""])
    for row in area_comparison:
        analysis_lines.append(
            "- `{}` `{}`m: real/original ratio `{}`, real mean `{}`, original mean `{}`".format(
                row["dataset"],
                int(float(row["distance_m"])),
                _fmt(row["real_div_original_mean"]),
                _fmt(row["real_mean"]),
                _fmt(row["original_mean"]),
            )
        )
    analysis_lines.extend(["", "## Simple Calibration on 20% of Nenrus", ""])
    for row in sorted(calibration_summary, key=lambda item: float(item["mae_mean"])):
        analysis_lines.append(
            "- `{}`: MAE `{}` +/- `{}`, mean relative error `{}`, within 10m `{}`".format(
                row["model"],
                _fmt(row["mae_mean"]),
                _fmt(row["mae_std"]),
                _fmt(row["mean_relative_error_mean"]),
                _fmt(row["within_10m_rate_mean"]),
            )
        )
    analysis_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The geometry comparison directly tests whether the model sees real-drone boxes as larger or smaller than same-distance original-dataset boxes.",
            "If real/original ratios are below 1.0, the real boxes are smaller than the original boxes at the same distance; since the model heavily relies on bbox size, that tends to push predictions upward.",
            "",
            "The calibration experiment is not a pure external test anymore, because it uses a small labelled subset of Nenrus.",
            "It answers a different question: if we are allowed a small calibration set from the real domain, can a simple mapping from predicted distance to true distance fix the systematic bias?",
            "",
        ]
    )
    analysis_md = REPORTS_DIR / "bbox_domain_and_calibration_analysis.md"
    analysis_md.write_text("\n".join(analysis_lines), encoding="utf-8")

    summary = {
        "geometry_rows_csv": str(REPORTS_DIR / "bbox_geometry_domain_rows.csv"),
        "bbox_width_distribution_csv": str(REPORTS_DIR / "bbox_width_norm_distribution_by_dataset_distance.csv"),
        "bbox_area_distribution_csv": str(REPORTS_DIR / "bbox_area_ratio_distribution_by_dataset_distance.csv"),
        "bbox_width_real_vs_original_csv": str(REPORTS_DIR / "bbox_width_norm_real_vs_original_shared_distances.csv"),
        "bbox_area_real_vs_original_csv": str(REPORTS_DIR / "bbox_area_ratio_real_vs_original_shared_distances.csv"),
        "simple_calibration_summary_csv": str(REPORTS_DIR / "simple_calibration_summary.csv"),
        "analysis_md": str(analysis_md),
        "plots": [
            str(PLOTS_DIR / "bbox_width_norm_original_vs_real_by_distance.png"),
            str(PLOTS_DIR / "bbox_area_ratio_original_vs_real_by_distance.png"),
            str(PLOTS_DIR / "bbox_width_norm_real_div_original_shared_distances.png"),
            str(PLOTS_DIR / "bbox_area_ratio_real_div_original_shared_distances.png"),
            str(PLOTS_DIR / "simple_calibration_test_performance.png"),
        ],
    }
    _write_json(REPORTS_DIR / "bbox_domain_and_calibration_summary.json", summary)
    return summary


if __name__ == "__main__":
    run()
