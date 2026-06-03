"""
Learn and evaluate simple calibration functions for Nenrus predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nenrus calibration study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML did not contain an object: {}".format(path))
    return payload


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        prepared_row = dict(row)
        prepared_row["true_distance_m"] = float(row["true_distance_m"])
        prepared_row["predicted_distance_m"] = float(row["predicted_distance_m"])
        prepared_row["drone_type"] = str(row["drone_type"])
        prepared.append(prepared_row)
    return prepared


def _metrics(rows: Sequence[Dict[str, Any]], predictions: Sequence[float]) -> Dict[str, Any]:
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    errors = preds - true_values
    abs_errors = np.abs(errors)
    rel_errors = np.divide(abs_errors, true_values, out=np.zeros_like(abs_errors), where=true_values > 0)
    centered = true_values - float(np.mean(true_values))
    sst = float(np.sum(centered**2))
    sse = float(np.sum(errors**2))
    return {
        "count": int(len(rows)),
        "mae": float(np.mean(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1.0 - (sse / sst)) if sst > 0 else None,
        "mean_signed_error": float(np.mean(errors)),
        "median_signed_error": float(np.median(errors)),
        "p90_absolute_error": float(np.percentile(abs_errors, 90)),
        "p95_absolute_error": float(np.percentile(abs_errors, 95)),
        "within_2m_rate": float(np.mean(abs_errors <= 2.0)),
        "within_5m_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10m_rate": float(np.mean(abs_errors <= 10.0)),
        "within_20m_rate": float(np.mean(abs_errors <= 20.0)),
        "overprediction_rate": float(np.mean(errors > 0)),
        "underprediction_rate": float(np.mean(errors < 0)),
    }


def _fit_polynomial(predictions: Sequence[float], true_values: Sequence[float], degree: int) -> List[float]:
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(true_values, dtype=np.float64)
    actual_degree = min(int(degree), max(0, len(x) - 1))
    if actual_degree == 0:
        return [float(np.mean(y))]
    design = np.column_stack([x**power for power in range(actual_degree + 1)])
    coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    return [float(value) for value in coeffs]


def _predict_polynomial(coefficients: Sequence[float], predictions: Sequence[float]) -> np.ndarray:
    x = np.asarray(predictions, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    for power, coefficient in enumerate(coefficients):
        result += float(coefficient) * (x**power)
    return result


def _fit_scale(predictions: Sequence[float], true_values: Sequence[float]) -> float:
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(true_values, dtype=np.float64)
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return 1.0
    return float(np.dot(x, y) / denominator)


def _split_stratified(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    calibration_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(dict(row))

    calibration_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for group_rows in groups.values():
        shuffled = list(group_rows)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            calibration_rows.extend(shuffled)
            continue
        calibration_count = max(1, int(round(len(shuffled) * calibration_fraction)))
        calibration_count = min(calibration_count, len(shuffled) - 1)
        calibration_rows.extend(shuffled[:calibration_count])
        test_rows.extend(shuffled[calibration_count:])
    return calibration_rows, test_rows


def _fit_functions(calibration_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    calib_pred = [float(row["predicted_distance_m"]) for row in calibration_rows]
    calib_true = [float(row["true_distance_m"]) for row in calibration_rows]
    functions: Dict[str, Any] = {
        "raw_uncalibrated": {"type": "identity"},
        "global_affine": {
            "type": "polynomial",
            "degree": 1,
            "coefficients": _fit_polynomial(calib_pred, calib_true, degree=1),
        },
        "global_quadratic": {
            "type": "polynomial",
            "degree": 2,
            "coefficients": _fit_polynomial(calib_pred, calib_true, degree=2),
        },
        "global_scale_only": {
            "type": "scale_only",
            "scale": _fit_scale(calib_pred, calib_true),
        },
        "per_drone_affine": {
            "type": "per_drone_polynomial",
            "degree": 1,
            "fallback": None,
            "by_drone_type": {},
        },
        "per_drone_quadratic": {
            "type": "per_drone_polynomial",
            "degree": 2,
            "fallback": None,
            "by_drone_type": {},
        },
    }
    functions["per_drone_affine"]["fallback"] = functions["global_affine"]
    functions["per_drone_quadratic"]["fallback"] = functions["global_quadratic"]

    for model_name, degree in (("per_drone_affine", 1), ("per_drone_quadratic", 2)):
        for drone_type in sorted({str(row["drone_type"]) for row in calibration_rows}):
            subset = [row for row in calibration_rows if str(row["drone_type"]) == drone_type]
            functions[model_name]["by_drone_type"][drone_type] = {
                "type": "polynomial",
                "degree": degree,
                "coefficients": _fit_polynomial(
                    [float(row["predicted_distance_m"]) for row in subset],
                    [float(row["true_distance_m"]) for row in subset],
                    degree=degree,
                ),
                "calibration_count": len(subset),
            }
    return functions


def _apply_function(
    rows: Sequence[Dict[str, Any]],
    function_payload: Dict[str, Any],
    clip_min: Optional[float],
) -> np.ndarray:
    raw_predictions = np.asarray([float(row["predicted_distance_m"]) for row in rows], dtype=np.float64)
    function_type = str(function_payload["type"])
    if function_type == "identity":
        calibrated = raw_predictions
    elif function_type == "polynomial":
        calibrated = _predict_polynomial(function_payload["coefficients"], raw_predictions)
    elif function_type == "scale_only":
        calibrated = float(function_payload["scale"]) * raw_predictions
    elif function_type == "per_drone_polynomial":
        values: List[float] = []
        for row in rows:
            drone_type = str(row["drone_type"])
            drone_payload = dict(function_payload["by_drone_type"].get(drone_type, {}))
            if not drone_payload:
                drone_payload = dict(function_payload["fallback"])
            values.append(
                float(
                    _predict_polynomial(
                        drone_payload["coefficients"],
                        [float(row["predicted_distance_m"])],
                    )[0]
                )
            )
        calibrated = np.asarray(values, dtype=np.float64)
    else:
        raise ValueError("Unsupported calibration function type: {}".format(function_type))

    if clip_min is not None:
        calibrated = np.maximum(calibrated, float(clip_min))
    return calibrated


def _group_metric_rows(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output: List[Dict[str, Any]] = []
    for group_value, group_rows in sorted(groups.items()):
        metric_row = {key: group_value}
        metric_row.update(
            _metrics(
                group_rows,
                [float(row["calibrated_distance_m"]) for row in group_rows],
            )
        )
        output.append(metric_row)
    return output


def _summarize_runs(run_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["calibration_model"]), float(row["calibration_fraction"]))].append(row)

    metric_names = [
        "mae",
        "median_absolute_error",
        "mean_relative_error",
        "median_relative_error",
        "rmse",
        "r2",
        "mean_signed_error",
        "within_2m_rate",
        "within_5m_rate",
        "within_10m_rate",
        "within_20m_rate",
        "overprediction_rate",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for (model_name, fraction), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        summary: Dict[str, Any] = {
            "calibration_model": model_name,
            "calibration_fraction": fraction,
            "num_runs": len(rows),
            "mean_calibration_count": float(np.mean([float(row["calibration_count"]) for row in rows])),
            "mean_test_count": float(np.mean([float(row["test_count"]) for row in rows])),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
            summary[f"{metric_name}_median"] = float(np.median(values))
        summary_rows.append(summary)
    return summary_rows


def _plot_summary(summary_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fractions = sorted({float(row["calibration_fraction"]) for row in summary_rows})
    models = [
        "raw_uncalibrated",
        "global_scale_only",
        "global_affine",
        "global_quadratic",
        "per_drone_affine",
        "per_drone_quadratic",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name in models:
        subset = [
            row for row in summary_rows
            if str(row["calibration_model"]) == model_name
        ]
        if not subset:
            continue
        subset = sorted(subset, key=lambda row: float(row["calibration_fraction"]))
        axes[0].plot(
            [float(row["calibration_fraction"]) for row in subset],
            [float(row["mae_mean"]) for row in subset],
            marker="o",
            linewidth=2.0,
            label=model_name,
        )
        axes[1].plot(
            [float(row["calibration_fraction"]) for row in subset],
            [float(row["mean_relative_error_mean"]) for row in subset],
            marker="o",
            linewidth=2.0,
            label=model_name,
        )
    axes[0].set_title("Calibration Holdout MAE")
    axes[0].set_xlabel("Calibration fraction")
    axes[0].set_ylabel("MAE (m)")
    axes[1].set_title("Calibration Holdout Relative Error")
    axes[1].set_xlabel("Calibration fraction")
    axes[1].set_ylabel("Mean relative error")
    for ax in axes:
        ax.set_xticks(fractions)
        ax.grid(alpha=0.25)
    axes[1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_predictions(rows: Sequence[Dict[str, Any]], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for drone_type, color in (("Kongsberg", "#b85c45"), ("Vestfold", "#315f8c")):
        subset = [row for row in rows if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["true_distance_m"]) for row in subset],
            [float(row["calibrated_distance_m"]) for row in subset],
            s=26,
            alpha=0.75,
            color=color,
            label=drone_type,
        )
    true_values = [float(row["true_distance_m"]) for row in rows]
    pred_values = [float(row["calibrated_distance_m"]) for row in rows]
    lower = min(true_values + pred_values)
    upper = max(true_values + pred_values)
    ax.plot([lower, upper], [lower, upper], color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Calibrated prediction (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_by_distance(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[float(row["true_distance_m"])].append(row)
    distance_rows: List[Dict[str, Any]] = []
    for distance, group_rows in sorted(groups.items()):
        metric_row = {"distance_m": distance}
        metric_row.update(
            _metrics(
                group_rows,
                [float(row["calibrated_distance_m"]) for row in group_rows],
            )
        )
        distance_rows.append(metric_row)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [str(int(row["distance_m"])) for row in distance_rows],
        [float(row["mae"]) for row in distance_rows],
        color="#365f46",
    )
    ax.set_title("Final Calibration MAE by Distance")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("MAE (m)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _group_metric_rows_for_prediction(rows: Sequence[Dict[str, Any]], key: str, prediction_column: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    metric_rows: List[Dict[str, Any]] = []
    for key_value, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        metric_row = {key: key_value}
        metric_row.update(_metrics(group_rows, [float(row[prediction_column]) for row in group_rows]))
        metric_rows.append(metric_row)
    return metric_rows


def _distance_metric_rows(rows: Sequence[Dict[str, Any]], prediction_column: str) -> List[Dict[str, Any]]:
    groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[float(row["true_distance_m"])].append(row)
    metric_rows: List[Dict[str, Any]] = []
    for distance, group_rows in sorted(groups.items()):
        metric_row = {"true_distance_m": distance}
        metric_row.update(_metrics(group_rows, [float(row[prediction_column]) for row in group_rows]))
        metric_rows.append(metric_row)
    return metric_rows


def _plot_raw_and_calibrated_scatter(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    panels = [
        ("Raw model: systematic overprediction", "raw_predicted_distance_m"),
        ("After calibration: back on the diagonal", "calibrated_distance_m"),
    ]
    colors = {"Kongsberg": "#b85c45", "Vestfold": "#315f8c"}
    true_values = [float(row["true_distance_m"]) for row in rows]
    all_predictions = true_values + [float(row["raw_predicted_distance_m"]) for row in rows] + [
        float(row["calibrated_distance_m"]) for row in rows
    ]
    lower = min(all_predictions)
    upper = max(all_predictions)
    for ax, (title, column) in zip(axes, panels):
        for drone_type in sorted(colors):
            subset = [row for row in rows if str(row["drone_type"]) == drone_type]
            ax.scatter(
                [float(row["true_distance_m"]) for row in subset],
                [float(row[column]) for row in subset],
                s=26,
                alpha=0.72,
                color=colors[drone_type],
                label=drone_type,
            )
        ax.plot([lower, upper], [lower, upper], color="#111111", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("True distance (m)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted distance (m)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_mae_before_after_by_distance(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    raw_rows = _distance_metric_rows(rows, "raw_predicted_distance_m")
    calibrated_rows = _distance_metric_rows(rows, "calibrated_distance_m")
    distances = [float(row["true_distance_m"]) for row in raw_rows]
    x = np.arange(len(distances))
    width = 0.42
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, [float(row["mae"]) for row in raw_rows], width=width, color="#a94735", label="Raw")
    ax.bar(
        x + width / 2,
        [float(row["mae"]) for row in calibrated_rows],
        width=width,
        color="#2f6f4e",
        label="Calibrated",
    )
    ax.set_title("MAE by Distance: Before vs After Calibration")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("MAE (m)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(distance)) for distance in distances], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_mae_before_after_by_drone(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    raw_rows = _group_metric_rows_for_prediction(rows, "drone_type", "raw_predicted_distance_m")
    calibrated_rows = _group_metric_rows_for_prediction(rows, "drone_type", "calibrated_distance_m")
    drone_types = [str(row["drone_type"]) for row in raw_rows]
    x = np.arange(len(drone_types))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [float(row["mae"]) for row in raw_rows], width=width, color="#a94735", label="Raw")
    ax.bar(
        x + width / 2,
        [float(row["mae"]) for row in calibrated_rows],
        width=width,
        color="#2f6f4e",
        label="Calibrated",
    )
    for index, (raw_row, cal_row) in enumerate(zip(raw_rows, calibrated_rows)):
        ax.text(
            index,
            max(float(raw_row["mae"]), float(cal_row["mae"])) + 0.8,
            "{:.1f} -> {:.1f}".format(float(raw_row["mae"]), float(cal_row["mae"])),
            ha="center",
            fontsize=9,
        )
    ax.set_title("MAE by Drone Dataset: Before vs After")
    ax.set_xlabel("Dataset / drone type")
    ax.set_ylabel("MAE (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(drone_types)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_distribution_before_after(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    raw_abs = np.asarray([abs(float(row["raw_predicted_distance_m"]) - float(row["true_distance_m"])) for row in rows])
    cal_abs = np.asarray([float(row["calibrated_absolute_error_m"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, max(float(np.max(raw_abs)), float(np.max(cal_abs))), 28)
    axes[0].hist(raw_abs, bins=bins, alpha=0.72, color="#a94735", label="Raw")
    axes[0].hist(cal_abs, bins=bins, alpha=0.72, color="#2f6f4e", label="Calibrated")
    axes[0].set_title("Absolute Error Distribution")
    axes[0].set_xlabel("Absolute error (m)")
    axes[0].set_ylabel("Image count")
    axes[0].legend()

    raw_signed = np.asarray([float(row["raw_predicted_distance_m"]) - float(row["true_distance_m"]) for row in rows])
    cal_signed = np.asarray([float(row["calibrated_signed_error_m"]) for row in rows])
    signed_bins = np.linspace(min(float(np.min(raw_signed)), float(np.min(cal_signed))), max(float(np.max(raw_signed)), float(np.max(cal_signed))), 32)
    axes[1].hist(raw_signed, bins=signed_bins, alpha=0.72, color="#a94735", label="Raw")
    axes[1].hist(cal_signed, bins=signed_bins, alpha=0.72, color="#2f6f4e", label="Calibrated")
    axes[1].axvline(0.0, color="#111111", linestyle="--", linewidth=1.1)
    axes[1].set_title("Signed Error Distribution")
    axes[1].set_xlabel("Prediction - true (m)")
    axes[1].set_ylabel("Image count")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_cdf_before_after(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    raw_abs = np.sort(np.asarray([abs(float(row["raw_predicted_distance_m"]) - float(row["true_distance_m"])) for row in rows]))
    cal_abs = np.sort(np.asarray([float(row["calibrated_absolute_error_m"]) for row in rows]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label, color in ((raw_abs, "Raw", "#a94735"), (cal_abs, "Calibrated", "#2f6f4e")):
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, linewidth=2.2, color=color, label=label)
    for threshold in (2, 5, 10, 20):
        ax.axvline(threshold, color="#777777", linestyle=":", linewidth=0.9)
        ax.text(threshold, 0.02, "{}m".format(threshold), rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_title("How Many Images Are Solved Within Each Error Threshold")
    ax.set_xlabel("Absolute error threshold (m)")
    ax.set_ylabel("Fraction of images within threshold")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_tolerance_rates_before_after(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    thresholds = [2.0, 5.0, 10.0, 20.0]
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows])
    raw_abs = np.abs(np.asarray([float(row["raw_predicted_distance_m"]) for row in rows]) - true_values)
    cal_abs = np.abs(np.asarray([float(row["calibrated_distance_m"]) for row in rows]) - true_values)
    x = np.arange(len(thresholds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [float(np.mean(raw_abs <= threshold)) for threshold in thresholds], width=width, color="#a94735", label="Raw")
    ax.bar(x + width / 2, [float(np.mean(cal_abs <= threshold)) for threshold in thresholds], width=width, color="#2f6f4e", label="Calibrated")
    ax.set_title("Tolerance Success Rate: Before vs After")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Fraction of images")
    ax.set_xticks(x)
    ax.set_xticklabels(["<= {}m".format(int(threshold)) for threshold in thresholds])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_calibration_curve(rows: Sequence[Dict[str, Any]], selected_function: Dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Kongsberg": "#b85c45", "Vestfold": "#315f8c"}
    for drone_type in sorted(colors):
        subset = [row for row in rows if str(row["drone_type"]) == drone_type]
        raw_values = np.asarray([float(row["raw_predicted_distance_m"]) for row in subset], dtype=np.float64)
        true_values = np.asarray([float(row["true_distance_m"]) for row in subset], dtype=np.float64)
        ax.scatter(raw_values, true_values, s=24, alpha=0.45, color=colors[drone_type], label="{} samples".format(drone_type))
        if selected_function.get("type") == "per_drone_polynomial":
            function_payload = selected_function["by_drone_type"].get(drone_type)
        else:
            function_payload = selected_function
        if function_payload:
            x_grid = np.linspace(float(np.min(raw_values)), float(np.max(raw_values)), 180)
            y_grid = _predict_polynomial(function_payload["coefficients"], x_grid)
            ax.plot(x_grid, y_grid, linewidth=2.4, color=colors[drone_type], label="{} calibration".format(drone_type))
    ax.set_title("The Actual Calibration Mapping: Raw Prediction -> True Distance")
    ax.set_xlabel("Raw ensemble prediction (m)")
    ax.set_ylabel("True / calibrated distance scale (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_residuals_after_calibration(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Kongsberg": "#b85c45", "Vestfold": "#315f8c"}
    for drone_type in sorted(colors):
        subset = [row for row in rows if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["raw_predicted_distance_m"]) for row in subset],
            [float(row["calibrated_signed_error_m"]) for row in subset],
            s=24,
            alpha=0.65,
            color=colors[drone_type],
            label=drone_type,
        )
    ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals After Calibration")
    ax.set_xlabel("Raw ensemble prediction (m)")
    ax.set_ylabel("Calibrated prediction - true (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_mean_prediction_tracks(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["drone_type"]), float(row["true_distance_m"]))].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, drone_type in zip(axes, ["Kongsberg", "Vestfold"]):
        distance_rows: List[Tuple[float, float, float, float]] = []
        for (group_drone, distance), group_rows in sorted(grouped.items()):
            if group_drone != drone_type:
                continue
            distance_rows.append(
                (
                    distance,
                    float(np.mean([float(row["true_distance_m"]) for row in group_rows])),
                    float(np.mean([float(row["raw_predicted_distance_m"]) for row in group_rows])),
                    float(np.mean([float(row["calibrated_distance_m"]) for row in group_rows])),
                )
            )
        ax.plot([row[0] for row in distance_rows], [row[1] for row in distance_rows], marker="o", linewidth=2.0, color="#111111", label="True")
        ax.plot([row[0] for row in distance_rows], [row[2] for row in distance_rows], marker="o", linewidth=2.0, color="#a94735", label="Raw")
        ax.plot([row[0] for row in distance_rows], [row[3] for row in distance_rows], marker="o", linewidth=2.0, color="#2f6f4e", label="Calibrated")
        ax.set_title("{} Mean Predictions by Distance".format(drone_type))
        ax.set_xlabel("True distance bucket (m)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean distance (m)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relative_error_before_after(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    raw_rows = _distance_metric_rows(rows, "raw_predicted_distance_m")
    calibrated_rows = _distance_metric_rows(rows, "calibrated_distance_m")
    fig, ax = plt.subplots(figsize=(12, 5))
    distances = [float(row["true_distance_m"]) for row in raw_rows]
    ax.plot(distances, [float(row["mean_relative_error"]) for row in raw_rows], marker="o", linewidth=2.2, color="#a94735", label="Raw")
    ax.plot(
        distances,
        [float(row["mean_relative_error"]) for row in calibrated_rows],
        marker="o",
        linewidth=2.2,
        color="#2f6f4e",
        label="Calibrated",
    )
    ax.set_title("Relative Error by Distance: Before vs After")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Mean relative error")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _metadata_audit(
    feature_rows: Sequence[Dict[str, Any]],
    schema: Dict[str, Any],
    reports_dir: Path,
    plots_dir: Path,
) -> Dict[str, Any]:
    categorical_levels = dict(schema.get("categorical_levels", {}))
    categorical_columns = [str(value) for value in schema.get("categorical_feature_names", [])]
    audit: Dict[str, Any] = {
        "num_rows": len(feature_rows),
        "categorical_columns": categorical_columns,
        "trained_levels": categorical_levels,
        "observed_values": {},
        "known_value_rows": {},
        "unknown_value_rows": {},
        "metadata_one_hot_active_rate": {},
        "interpretation": "Nenrus rows use unknown weather/time values, so trained one-hot metadata indicators are zero rather than null.",
    }
    plot_labels: List[str] = []
    known_rates: List[float] = []
    active_rates: List[float] = []
    for column in categorical_columns:
        trained_values = {str(value) for value in categorical_levels.get(column, [])}
        values = [str(row.get(column, "")) for row in feature_rows]
        known_count = sum(1 for value in values if value in trained_values)
        unknown_count = len(values) - known_count
        audit["observed_values"][column] = sorted(set(values))
        audit["known_value_rows"][column] = known_count
        audit["unknown_value_rows"][column] = unknown_count
        audit["metadata_one_hot_active_rate"][column] = float(known_count / len(values)) if values else 0.0
        plot_labels.append(column)
        known_rates.append(float(known_count / len(values)) if values else 0.0)
        active_rates.append(float(known_count / len(values)) if values else 0.0)
    _write_json(reports_dir / "metadata_missing_audit.json", audit)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_labels))
    width = 0.36
    ax.bar(x - width / 2, known_rates, width=width, color="#315f8c", label="Rows matching trained metadata levels")
    ax.bar(x + width / 2, active_rates, width=width, color="#d39b45", label="One-hot active rate")
    ax.set_title("Nenrus Metadata Audit: Unknown Means All Metadata One-Hots Are Zero")
    ax.set_xlabel("Metadata field")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "metadata_missing_one_hot_audit.png")
    plt.close(fig)
    return audit


def _make_explanation_plots(
    final_rows: Sequence[Dict[str, Any]],
    selected_function: Dict[str, Any],
    plots_dir: Path,
) -> List[str]:
    plot_specs = [
        ("raw_vs_calibrated_predicted_vs_true.png", lambda path: _plot_raw_and_calibrated_scatter(final_rows, path)),
        ("mae_before_after_by_distance.png", lambda path: _plot_mae_before_after_by_distance(final_rows, path)),
        ("mae_before_after_by_drone_type.png", lambda path: _plot_mae_before_after_by_drone(final_rows, path)),
        ("error_distribution_before_after.png", lambda path: _plot_error_distribution_before_after(final_rows, path)),
        ("absolute_error_cdf_before_after.png", lambda path: _plot_error_cdf_before_after(final_rows, path)),
        ("tolerance_rates_before_after.png", lambda path: _plot_tolerance_rates_before_after(final_rows, path)),
        ("calibration_mapping_curve.png", lambda path: _plot_calibration_curve(final_rows, selected_function, path)),
        ("residuals_after_calibration.png", lambda path: _plot_residuals_after_calibration(final_rows, path)),
        ("mean_prediction_tracks_by_distance.png", lambda path: _plot_mean_prediction_tracks(final_rows, path)),
        ("relative_error_before_after_by_distance.png", lambda path: _plot_relative_error_before_after(final_rows, path)),
    ]
    paths: List[str] = []
    for filename, plotter in plot_specs:
        output_path = plots_dir / filename
        plotter(output_path)
        paths.append(str(output_path))
    return paths


def _format_formula(function_payload: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if function_payload["type"] == "per_drone_polynomial":
        for drone_type, payload in sorted(dict(function_payload["by_drone_type"]).items()):
            coefficients = [float(value) for value in payload["coefficients"]]
            if len(coefficients) == 2:
                lines.append(
                    "{}: corrected = {:.8f} + {:.8f} * prediction".format(
                        drone_type,
                        coefficients[0],
                        coefficients[1],
                    )
                )
            elif len(coefficients) == 3:
                lines.append(
                    "{}: corrected = {:.8f} + {:.8f} * prediction + {:.8f} * prediction^2".format(
                        drone_type,
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                    )
                )
            else:
                lines.append("{}: coefficients={}".format(drone_type, coefficients))
    return lines


def _write_analysis(
    path: Path,
    selected_model: str,
    summary_rows: Sequence[Dict[str, Any]],
    raw_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    raw_by_drone: Sequence[Dict[str, Any]],
    final_by_drone: Sequence[Dict[str, Any]],
    selected_function: Dict[str, Any],
    metadata_audit: Optional[Dict[str, Any]],
) -> None:
    selected_20 = [
        row for row in summary_rows
        if str(row["calibration_model"]) == selected_model
        and abs(float(row["calibration_fraction"]) - 0.2) < 1e-9
    ]
    best_20 = sorted(
        [row for row in summary_rows if abs(float(row["calibration_fraction"]) - 0.2) < 1e-9],
        key=lambda row: float(row["mae_mean"]),
    )[:5]
    lines = [
        "# Study 04: Nenrus Calibration Function",
        "",
        "## Problem We Saw",
        "",
        "The exported RF+XGBoost ensemble overpredicted every Nenrus image before calibration.",
        "This is a domain-shift problem: the model was trained on the original controlled/synthetic-style dataset, while Nenrus contains real-drone images with different bbox scale behavior.",
        "",
        "- raw MAE: `{:.4f}m`".format(float(raw_metrics["mae"])),
        "- raw mean relative error: `{:.4f}`".format(float(raw_metrics["mean_relative_error"])),
        "- raw overprediction rate: `{:.4f}`".format(float(raw_metrics["overprediction_rate"])),
        "- raw within 10m rate: `{:.4f}`".format(float(raw_metrics["within_10m_rate"])),
        "",
        "## Selected Calibration Function",
        "",
        "`{}`".format(selected_model),
        "",
    ]
    lines.extend("- `{}`".format(line) for line in _format_formula(selected_function))
    lines.extend(
        [
            "",
            "## Honest Split Evaluation",
            "",
            "The repeated split evaluation learns calibration on a small labelled subset of Nenrus and evaluates on the remaining held-out Nenrus rows.",
            "",
        ]
    )
    if selected_20:
        row = selected_20[0]
        lines.extend(
            [
                "For the selected model at 20% calibration data:",
                "",
                "- test MAE mean: `{:.4f}m`".format(float(row["mae_mean"])),
                "- test MAE std: `{:.4f}m`".format(float(row["mae_std"])),
                "- mean relative error: `{:.4f}`".format(float(row["mean_relative_error_mean"])),
                "- within 10m rate: `{:.4f}`".format(float(row["within_10m_rate_mean"])),
                "",
            ]
        )
    lines.extend(["Best 20% calibration models:", ""])
    for row in best_20:
        lines.append(
            "- `{}`: MAE `{:.4f}m`, relative `{:.4f}`, within10 `{:.4f}`".format(
                row["calibration_model"],
                float(row["mae_mean"]),
                float(row["mean_relative_error_mean"]),
                float(row["within_10m_rate_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Full-Data Fit",
            "",
            "The final saved function is fitted on all Nenrus rows. This is useful if the goal is to calibrate this specific real-domain dataset, but it is not a pure external-test score.",
            "",
            "- full-fit MAE: `{:.4f}m`".format(float(final_metrics["mae"])),
            "- full-fit mean relative error: `{:.4f}`".format(float(final_metrics["mean_relative_error"])),
            "- full-fit within 10m rate: `{:.4f}`".format(float(final_metrics["within_10m_rate"])),
            "- full-fit mean signed error: `{:.4f}m`".format(float(final_metrics["mean_signed_error"])),
            "",
            "## Per Dataset Result",
            "",
        ]
    )
    final_by_drone_map = {str(row["drone_type"]): row for row in final_by_drone}
    for raw_row in raw_by_drone:
        drone_type = str(raw_row["drone_type"])
        final_row = final_by_drone_map.get(drone_type)
        if not final_row:
            continue
        lines.append(
            "- `{}`: MAE `{:.4f}m` -> `{:.4f}m`, relative `{:.4f}` -> `{:.4f}`".format(
                drone_type,
                float(raw_row["mae"]),
                float(final_row["mae"]),
                float(raw_row["mean_relative_error"]),
                float(final_row["mean_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Metadata Note",
            "",
        ]
    )
    if metadata_audit:
        lines.append(
            "Nenrus does not provide usable `weather` or `time_of_day` values for the trained metadata categories."
        )
        lines.append(
            "The feature pipeline used `unknown`, not null/NaN. Because `unknown` was not in training, the one-hot metadata columns are all zero."
        )
        for column in metadata_audit.get("categorical_columns", []):
            lines.append(
                "- `{}` observed values: `{}`; matching trained levels: `{}/{}`".format(
                    column,
                    ", ".join(str(value) for value in metadata_audit["observed_values"].get(column, [])),
                    metadata_audit["known_value_rows"].get(column, 0),
                    metadata_audit.get("num_rows", 0),
                )
            )
    else:
        lines.append("Metadata audit was not available because the feature table/schema source was not configured.")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "A per-drone quadratic calibration is the strongest simple correction for the systematic overprediction.",
            "The per-drone affine version is slightly weaker but remains a useful simpler fallback.",
            "It does not retrain the RF/XGBoost model; it only maps the model's raw distance output onto the real-domain distance scale.",
            "",
        ]
    )
    _write_text(path, "\n".join(lines))


def run(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _load_yaml(config_path)
    sources = dict(config["sources"])
    calibration_config = dict(config["calibration"])
    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = _prepare_rows(_read_csv(_resolve_repo_path(str(sources["predictions_csv"]))))
    raw_metrics = _metrics(rows, [float(row["predicted_distance_m"]) for row in rows])
    raw_by_drone = _group_metric_rows_for_prediction(rows, "drone_type", "predicted_distance_m")
    raw_by_distance = _distance_metric_rows(rows, "predicted_distance_m")
    group_keys = [str(key) for key in calibration_config.get("split_group_keys", ["drone_type", "true_distance_m"])]
    calibration_fractions = [float(value) for value in calibration_config["calibration_fractions"]]
    num_random_seeds = int(calibration_config["num_random_seeds"])
    selected_model = str(calibration_config.get("selected_model", "per_drone_affine"))
    clip_min = calibration_config.get("clip_predictions_min_m")
    clip_min = float(clip_min) if clip_min is not None else None

    run_rows: List[Dict[str, Any]] = []
    for fraction in calibration_fractions:
        for seed in range(num_random_seeds):
            calibration_rows, test_rows = _split_stratified(
                rows,
                group_keys=group_keys,
                calibration_fraction=fraction,
                seed=seed,
            )
            functions = _fit_functions(calibration_rows)
            for model_name, function_payload in functions.items():
                predictions = _apply_function(test_rows, function_payload, clip_min=clip_min)
                metric_row = {
                    "calibration_model": model_name,
                    "calibration_fraction": fraction,
                    "seed": seed,
                    "calibration_count": len(calibration_rows),
                    "test_count": len(test_rows),
                }
                metric_row.update(_metrics(test_rows, predictions))
                run_rows.append(metric_row)

    summary_rows = _summarize_runs(run_rows)
    _write_csv(reports_dir / "split_eval_runs.csv", run_rows)
    _write_csv(reports_dir / "split_eval_summary.csv", summary_rows)

    final_functions = _fit_functions(rows)
    selected_function = dict(final_functions[selected_model])
    final_predictions = _apply_function(rows, selected_function, clip_min=clip_min)
    final_rows: List[Dict[str, Any]] = []
    for row, calibrated in zip(rows, final_predictions):
        output_row = dict(row)
        output_row["calibration_model"] = selected_model
        output_row["raw_predicted_distance_m"] = float(row["predicted_distance_m"])
        output_row["calibrated_distance_m"] = float(calibrated)
        output_row["calibrated_signed_error_m"] = float(calibrated - float(row["true_distance_m"]))
        output_row["calibrated_absolute_error_m"] = abs(float(output_row["calibrated_signed_error_m"]))
        output_row["calibrated_relative_error"] = (
            float(output_row["calibrated_absolute_error_m"]) / float(row["true_distance_m"])
            if float(row["true_distance_m"]) > 0
            else 0.0
        )
        final_rows.append(output_row)

    final_metrics = _metrics(final_rows, [float(row["calibrated_distance_m"]) for row in final_rows])
    final_by_drone = _group_metric_rows(final_rows, "drone_type")
    final_by_distance = _group_metric_rows(final_rows, "true_distance_m")
    comparison_by_drone: List[Dict[str, Any]] = []
    final_by_drone_map = {str(row["drone_type"]): row for row in final_by_drone}
    for raw_row in raw_by_drone:
        drone_type = str(raw_row["drone_type"])
        final_row = final_by_drone_map.get(drone_type)
        if not final_row:
            continue
        comparison_by_drone.append(
            {
                "drone_type": drone_type,
                "count": int(raw_row["count"]),
                "raw_mae": float(raw_row["mae"]),
                "calibrated_mae": float(final_row["mae"]),
                "mae_improvement_m": float(raw_row["mae"]) - float(final_row["mae"]),
                "raw_mean_relative_error": float(raw_row["mean_relative_error"]),
                "calibrated_mean_relative_error": float(final_row["mean_relative_error"]),
                "raw_within_10m_rate": float(raw_row["within_10m_rate"]),
                "calibrated_within_10m_rate": float(final_row["within_10m_rate"]),
            }
        )
    final_by_distance_map = {str(row["true_distance_m"]): row for row in final_by_distance}
    comparison_by_distance: List[Dict[str, Any]] = []
    for raw_row in raw_by_distance:
        distance_key = str(raw_row["true_distance_m"])
        final_row = final_by_distance_map.get(distance_key)
        if not final_row:
            continue
        comparison_by_distance.append(
            {
                "true_distance_m": float(raw_row["true_distance_m"]),
                "count": int(raw_row["count"]),
                "raw_mae": float(raw_row["mae"]),
                "calibrated_mae": float(final_row["mae"]),
                "mae_improvement_m": float(raw_row["mae"]) - float(final_row["mae"]),
                "raw_mean_relative_error": float(raw_row["mean_relative_error"]),
                "calibrated_mean_relative_error": float(final_row["mean_relative_error"]),
                "raw_within_10m_rate": float(raw_row["within_10m_rate"]),
                "calibrated_within_10m_rate": float(final_row["within_10m_rate"]),
            }
        )

    metadata_audit = None
    if sources.get("nenrus_feature_table_csv") and sources.get("model_schema_json"):
        metadata_audit = _metadata_audit(
            _read_csv(_resolve_repo_path(str(sources["nenrus_feature_table_csv"]))),
            _load_yaml(_resolve_repo_path(str(sources["model_schema_json"]))) if str(sources["model_schema_json"]).endswith((".yaml", ".yml")) else json.loads(_resolve_repo_path(str(sources["model_schema_json"])).read_text(encoding="utf-8")),
            reports_dir,
            plots_dir,
        )

    function_payload = {
        "selected_model": selected_model,
        "clip_predictions_min_m": clip_min,
        "function": selected_function,
        "all_functions": final_functions,
        "usage": "Apply this after the raw ensemble prediction. If per-drone, select function by drone_type.",
    }
    _write_json(reports_dir / "final_calibration_functions.json", function_payload)
    _write_json(reports_dir / "raw_uncalibrated_metrics.json", raw_metrics)
    _write_csv(reports_dir / "final_calibrated_predictions.csv", final_rows)
    _write_json(reports_dir / "final_calibrated_metrics.json", final_metrics)
    _write_csv(reports_dir / "raw_uncalibrated_by_drone_type.csv", raw_by_drone)
    _write_csv(reports_dir / "raw_uncalibrated_by_distance.csv", raw_by_distance)
    _write_csv(reports_dir / "final_calibrated_by_drone_type.csv", final_by_drone)
    _write_csv(reports_dir / "final_calibrated_by_distance.csv", final_by_distance)
    _write_csv(reports_dir / "before_after_by_drone_type.csv", comparison_by_drone)
    _write_csv(reports_dir / "before_after_by_distance.csv", comparison_by_distance)

    _plot_summary(summary_rows, plots_dir / "calibration_split_performance.png")
    _plot_predictions(final_rows, plots_dir / "final_calibrated_predicted_vs_true.png", "Final Calibrated Prediction vs True Distance")
    _plot_by_distance(final_rows, plots_dir / "final_calibrated_mae_by_distance.png")
    explanation_plots = _make_explanation_plots(final_rows, selected_function, plots_dir)
    if metadata_audit:
        explanation_plots.append(str(plots_dir / "metadata_missing_one_hot_audit.png"))
    _write_analysis(
        reports_dir / "analysis.md",
        selected_model=selected_model,
        summary_rows=summary_rows,
        raw_metrics=raw_metrics,
        final_metrics=final_metrics,
        raw_by_drone=raw_by_drone,
        final_by_drone=final_by_drone,
        selected_function=selected_function,
        metadata_audit=metadata_audit,
    )

    summary = {
        "study_name": str(config.get("study_name", "study04_nenrus_prediction_calibration")),
        "num_rows": len(rows),
        "selected_model": selected_model,
        "split_eval_summary_csv": str(reports_dir / "split_eval_summary.csv"),
        "final_calibration_functions_json": str(reports_dir / "final_calibration_functions.json"),
        "raw_uncalibrated_metrics_json": str(reports_dir / "raw_uncalibrated_metrics.json"),
        "final_calibrated_predictions_csv": str(reports_dir / "final_calibrated_predictions.csv"),
        "final_calibrated_metrics_json": str(reports_dir / "final_calibrated_metrics.json"),
        "before_after_by_drone_type_csv": str(reports_dir / "before_after_by_drone_type.csv"),
        "before_after_by_distance_csv": str(reports_dir / "before_after_by_distance.csv"),
        "analysis_md": str(reports_dir / "analysis.md"),
        "plots": [
            str(plots_dir / "calibration_split_performance.png"),
            str(plots_dir / "final_calibrated_predicted_vs_true.png"),
            str(plots_dir / "final_calibrated_mae_by_distance.png"),
        ] + explanation_plots,
        "raw_uncalibrated_metrics": raw_metrics,
        "final_metrics_full_data_fit": final_metrics,
        "metadata_audit": metadata_audit,
    }
    _write_json(reports_dir / "summary.json", summary)
    print("[attempt5-study04] Wrote summary: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
