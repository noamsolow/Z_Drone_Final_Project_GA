"""
Evaluate the exported Attempt 5 RF+XGBoost ensemble on the prepared Nenrus dataset.

The external dataset is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study02.run_study_02 import _build_rf_matrix as _build_tabular_matrix
from attempts.attempt3.studies.study03.run_study_03 import _build_geometry_feature_dict
from attempts.attempt3.studies.study01.run_study_01 import _write_csv_rows, _write_json
from pipeline.data.loader import yolo_normalized_to_absolute_bbox
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import (
    ContextWindowSpec,
    compute_depth_representation,
    extract_context_crop,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported Attempt 5 ensemble on Nenrus.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Study 02 config.",
    )
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Ignore any cached feature table and recompute depth features.",
    )
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("YAML did not contain an object: {}".format(path))
    return loaded


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _round_float(value: float, digits: int = 10) -> float:
    return float(round(float(value), digits))


def _parse_yolo_label(label_path: Path, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty label file: {}".format(label_path))

    parsed_boxes: List[Tuple[int, int, int, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = parts
        parsed_boxes.append(
            yolo_normalized_to_absolute_bbox(
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
                image_width=image_width,
                image_height=image_height,
            )
        )
    if not parsed_boxes:
        raise ValueError("No valid YOLO bbox rows in: {}".format(label_path))

    return max(parsed_boxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _drone_type_from_manifest_row(row: Dict[str, Any]) -> str:
    explicit = str(row.get("drone_type", "")).strip()
    if explicit:
        return explicit
    image_path = str(row.get("image_path", ""))
    for candidate in ("Kongsberg", "Vestfold"):
        if candidate.lower() in image_path.lower():
            return candidate
    return "unknown"


def _distance_from_manifest_row(row: Dict[str, Any]) -> float:
    value = row.get("distance_m", row.get("distance", ""))
    return float(value)


def _build_feature_rows(
    manifest_rows: Sequence[Dict[str, Any]],
    depth_model: Any,
    raw_depth_feature_name: str,
    context_spec: ContextWindowSpec,
    aggregation_method: str,
    missing_weather_value: str,
    missing_time_of_day_value: str,
    progress_log_every: int,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows, start=1):
        image_path = Path(str(manifest_row["image_path"]))
        label_path = Path(str(manifest_row["label_path"]))
        drone_type = _drone_type_from_manifest_row(manifest_row)
        true_distance = _distance_from_manifest_row(manifest_row)

        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        image_width, image_height = image.size
        bbox = _parse_yolo_label(label_path, image_width=image_width, image_height=image_height)

        geometry_features = _build_geometry_feature_dict(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )
        crop_data = extract_context_crop(
            image=image,
            bbox=bbox,
            context_spec=context_spec,
        )
        depth_map = estimate_relative_depth(crop_data["image"], depth_model)
        representation = compute_depth_representation(
            depth_map=depth_map,
            bbox=crop_data["local_bbox"],
            aggregation_method=aggregation_method,
        )

        row: Dict[str, Any] = {
            "sample_id": "{}::{}".format(drone_type, image_path.name),
            "drone_type": drone_type,
            "distance_m": _round_float(true_distance),
            "true_distance_m": _round_float(true_distance),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "bbox_x1": int(bbox[0]),
            "bbox_y1": int(bbox[1]),
            "bbox_x2": int(bbox[2]),
            "bbox_y2": int(bbox[3]),
            "weather": missing_weather_value,
            "time_of_day": missing_time_of_day_value,
            raw_depth_feature_name: _round_float(float(representation["object_depth"])),
        }
        for feature_name, value in geometry_features.items():
            row[feature_name] = _round_float(float(value))
        output_rows.append(row)

        if index == 1 or index == len(manifest_rows) or index % progress_log_every == 0:
            print(
                "[attempt5-study02] Features {}/{} | {} | distance={}m".format(
                    index,
                    len(manifest_rows),
                    drone_type,
                    true_distance,
                )
            )
    return output_rows


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    errors = np.asarray([float(row["signed_error_m"]) for row in rows], dtype=np.float64)
    abs_errors = np.abs(errors)
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    pred_values = np.asarray([float(row["predicted_distance_m"]) for row in rows], dtype=np.float64)
    rel_errors = np.divide(abs_errors, true_values, out=np.zeros_like(abs_errors), where=true_values > 0)
    if len(rows) == 0:
        return {
            "count": 0,
            "mae": None,
            "median_absolute_error": None,
            "mean_relative_error": None,
            "rmse": None,
            "r2": None,
        }
    sse = float(np.sum((pred_values - true_values) ** 2))
    centered = true_values - float(np.mean(true_values))
    sst = float(np.sum(centered ** 2))
    return {
        "count": int(len(rows)),
        "mae": float(np.mean(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "rmse": float(math.sqrt(float(np.mean(errors**2)))),
        "r2": float(1.0 - (sse / sst)) if sst > 0 else None,
        "mean_signed_error": float(np.mean(errors)),
        "median_signed_error": float(np.median(errors)),
        "p90_absolute_error": float(np.percentile(abs_errors, 90)),
        "p95_absolute_error": float(np.percentile(abs_errors, 95)),
        "max_absolute_error": float(np.max(abs_errors)),
        "within_2m_rate": float(np.mean(abs_errors <= 2.0)),
        "within_5m_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10m_rate": float(np.mean(abs_errors <= 10.0)),
        "within_20m_rate": float(np.mean(abs_errors <= 20.0)),
        "overprediction_rate": float(np.mean(errors > 0)),
        "underprediction_rate": float(np.mean(errors < 0)),
        "mean_true_distance_m": float(np.mean(true_values)),
        "mean_predicted_distance_m": float(np.mean(pred_values)),
    }


def _group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return grouped


def _group_metric_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for key_values, group in sorted(_group_rows(rows, keys).items(), key=lambda item: item[0]):
        metric_values = _metrics(group)
        output_row = {key: value for key, value in zip(keys, key_values)}
        output_row.update(metric_values)
        output.append(output_row)
    return output


def _prediction_rows(
    feature_rows: Sequence[Dict[str, Any]],
    rf_predictions: Sequence[float],
    xgb_predictions: Sequence[float],
    ensemble_predictions: Sequence[float],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row, rf_pred, xgb_pred, pred in zip(feature_rows, rf_predictions, xgb_predictions, ensemble_predictions):
        true_distance = float(row["true_distance_m"])
        signed_error = float(pred - true_distance)
        output.append(
            {
                "sample_id": row["sample_id"],
                "drone_type": row["drone_type"],
                "distance_m": float(row["distance_m"]),
                "true_distance_m": true_distance,
                "predicted_distance_m": float(pred),
                "rf_predicted_distance_m": float(rf_pred),
                "xgb_predicted_distance_m": float(xgb_pred),
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "bbox_width_norm": float(row["bbox_width_norm"]),
                "bbox_height_norm": float(row["bbox_height_norm"]),
                "bbox_area_ratio": float(row["bbox_area_ratio"]),
                "raw_depth_feature": float(row.get("bbox_only__inner50_median__object_depth", 0.0)),
                "rf_xgb_prediction_delta_m": float(rf_pred - xgb_pred),
            }
        )
    return output


def _safe_metric(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None and value != "" else float("nan")


def _save_bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#2f6f6d",
    rotate: int = 45,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=rotate)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _make_plots(
    predictions: Sequence[Dict[str, Any]],
    by_distance: Sequence[Dict[str, Any]],
    by_drone: Sequence[Dict[str, Any]],
    by_drone_distance: Sequence[Dict[str, Any]],
    plots_dir: Path,
    dpi: int,
) -> List[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    old_dpi = plt.rcParams.get("figure.dpi", 100)
    plt.rcParams["figure.dpi"] = dpi
    output_paths: List[str] = []

    true_values = np.asarray([float(row["true_distance_m"]) for row in predictions], dtype=np.float64)
    pred_values = np.asarray([float(row["predicted_distance_m"]) for row in predictions], dtype=np.float64)
    signed_errors = np.asarray([float(row["signed_error_m"]) for row in predictions], dtype=np.float64)
    abs_errors = np.abs(signed_errors)
    drone_types = sorted({str(row["drone_type"]) for row in predictions})
    colors = {"Kongsberg": "#bf5b45", "Vestfold": "#315f8c", "unknown": "#888888"}

    path = plots_dir / "predicted_vs_true_by_drone_type.png"
    fig, ax = plt.subplots(figsize=(7, 7))
    for drone_type in drone_types:
        subset = [row for row in predictions if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["true_distance_m"]) for row in subset],
            [float(row["predicted_distance_m"]) for row in subset],
            s=24,
            alpha=0.72,
            label=drone_type,
            color=colors.get(drone_type, "#666666"),
        )
    min_axis = min(float(np.min(true_values)), float(np.min(pred_values)))
    max_axis = max(float(np.max(true_values)), float(np.max(pred_values)))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], color="#111111", linewidth=1.4, linestyle="--")
    ax.set_title("Predicted vs True Distance by Drone Type")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Predicted distance (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "signed_error_vs_distance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    for drone_type in drone_types:
        subset = [row for row in predictions if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["true_distance_m"]) for row in subset],
            [float(row["signed_error_m"]) for row in subset],
            s=24,
            alpha=0.72,
            label=drone_type,
            color=colors.get(drone_type, "#666666"),
        )
    ax.axhline(0.0, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_title("Signed Error vs True Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Prediction - GT (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    distance_rows = sorted(by_distance, key=lambda row: float(row["distance_m"]))
    labels = [str(int(float(row["distance_m"]))) for row in distance_rows]
    path = plots_dir / "mae_by_distance.png"
    _save_bar_plot(
        labels,
        [_safe_metric(row, "mae") for row in distance_rows],
        "MAE by Distance",
        "MAE (m)",
        path,
        color="#365f46",
    )
    output_paths.append(str(path))

    path = plots_dir / "mean_signed_error_by_distance.png"
    _save_bar_plot(
        labels,
        [_safe_metric(row, "mean_signed_error") for row in distance_rows],
        "Mean Signed Error by Distance",
        "Prediction - GT (m)",
        path,
        color="#8a5135",
    )
    output_paths.append(str(path))

    path = plots_dir / "mean_relative_error_by_distance.png"
    _save_bar_plot(
        labels,
        [_safe_metric(row, "mean_relative_error") for row in distance_rows],
        "Mean Relative Error by Distance",
        "Relative error",
        path,
        color="#4c6b8a",
    )
    output_paths.append(str(path))

    drone_rows = sorted(by_drone, key=lambda row: str(row["drone_type"]))
    path = plots_dir / "mae_by_drone_type.png"
    _save_bar_plot(
        [str(row["drone_type"]) for row in drone_rows],
        [_safe_metric(row, "mae") for row in drone_rows],
        "MAE by Drone Type / Source",
        "MAE (m)",
        path,
        color="#8b6f2a",
        rotate=0,
    )
    output_paths.append(str(path))

    path = plots_dir / "absolute_error_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abs_errors, bins=28, color="#385f7d", alpha=0.86)
    ax.set_title("Absolute Error Distribution")
    ax.set_xlabel("Absolute error (m)")
    ax.set_ylabel("Image count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "signed_error_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(signed_errors, bins=32, color="#844a4a", alpha=0.86)
    ax.axvline(0.0, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_title("Signed Error Distribution")
    ax.set_xlabel("Prediction - GT (m)")
    ax.set_ylabel("Image count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "absolute_error_boxplot_by_drone_type.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_abs = [
        [float(row["absolute_error_m"]) for row in predictions if str(row["drone_type"]) == drone_type]
        for drone_type in drone_types
    ]
    ax.boxplot(grouped_abs, tick_labels=drone_types, showfliers=True)
    ax.set_title("Absolute Error by Drone Type")
    ax.set_ylabel("Absolute error (m)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "mean_true_vs_predicted_by_distance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    distances = [float(row["distance_m"]) for row in distance_rows]
    ax.plot(distances, [float(row["mean_true_distance_m"]) for row in distance_rows], marker="o", label="GT")
    ax.plot(distances, [float(row["mean_predicted_distance_m"]) for row in distance_rows], marker="o", label="Prediction")
    ax.set_title("Mean GT vs Mean Prediction by Distance")
    ax.set_xlabel("Distance bucket (m)")
    ax.set_ylabel("Distance (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "within_tolerance_by_distance.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.26
    x = np.arange(len(distance_rows))
    ax.bar(x - width, [float(row["within_5m_rate"]) for row in distance_rows], width, label="<=5m")
    ax.bar(x, [float(row["within_10m_rate"]) for row in distance_rows], width, label="<=10m")
    ax.bar(x + width, [float(row["within_20m_rate"]) for row in distance_rows], width, label="<=20m")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Within-Tolerance Rates by Distance")
    ax.set_xlabel("Distance bucket (m)")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "mae_heatmap_drone_type_distance.png"
    heatmap_distances = sorted({float(row["distance_m"]) for row in by_drone_distance})
    heatmap_drones = sorted({str(row["drone_type"]) for row in by_drone_distance})
    value_lookup = {
        (str(row["drone_type"]), float(row["distance_m"])): float(row["mae"])
        for row in by_drone_distance
    }
    matrix = np.asarray(
        [
            [value_lookup.get((drone_type, distance), np.nan) for distance in heatmap_distances]
            for drone_type in heatmap_drones
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(11, max(3, len(heatmap_drones) * 1.4)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(heatmap_distances)))
    ax.set_xticklabels([str(int(distance)) for distance in heatmap_distances], rotation=45)
    ax.set_yticks(np.arange(len(heatmap_drones)))
    ax.set_yticklabels(heatmap_drones)
    ax.set_title("MAE Heatmap by Drone Type and Distance")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Drone type / source")
    fig.colorbar(image, ax=ax, label="MAE (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "bbox_width_norm_vs_distance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    for drone_type in drone_types:
        subset = [row for row in predictions if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["true_distance_m"]) for row in subset],
            [float(row["bbox_width_norm"]) for row in subset],
            s=24,
            alpha=0.72,
            label=drone_type,
            color=colors.get(drone_type, "#666666"),
        )
    ax.set_title("BBox Width Ratio vs Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("bbox_width_norm")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "raw_depth_feature_vs_distance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    for drone_type in drone_types:
        subset = [row for row in predictions if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["true_distance_m"]) for row in subset],
            [float(row["raw_depth_feature"]) for row in subset],
            s=24,
            alpha=0.72,
            label=drone_type,
            color=colors.get(drone_type, "#666666"),
        )
    ax.set_title("Raw Relative-Depth Feature vs Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("bbox_only inner50 median depth")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "rf_vs_xgb_predictions.png"
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        [float(row["rf_predicted_distance_m"]) for row in predictions],
        [float(row["xgb_predicted_distance_m"]) for row in predictions],
        s=22,
        alpha=0.72,
        color="#566b36",
    )
    min_pred = min(
        min(float(row["rf_predicted_distance_m"]) for row in predictions),
        min(float(row["xgb_predicted_distance_m"]) for row in predictions),
    )
    max_pred = max(
        max(float(row["rf_predicted_distance_m"]) for row in predictions),
        max(float(row["xgb_predicted_distance_m"]) for row in predictions),
    )
    ax.plot([min_pred, max_pred], [min_pred, max_pred], color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("RF vs XGBoost Predictions")
    ax.set_xlabel("RF prediction (m)")
    ax.set_ylabel("XGBoost prediction (m)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "rf_xgb_prediction_delta_by_distance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        [float(row["true_distance_m"]) for row in predictions],
        [float(row["rf_xgb_prediction_delta_m"]) for row in predictions],
        s=22,
        alpha=0.72,
        color="#6b4f7d",
    )
    ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("RF - XGBoost Prediction Difference by Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("RF prediction - XGB prediction (m)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "absolute_error_vs_bbox_area_ratio.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    for drone_type in drone_types:
        subset = [row for row in predictions if str(row["drone_type"]) == drone_type]
        ax.scatter(
            [float(row["bbox_area_ratio"]) for row in subset],
            [float(row["absolute_error_m"]) for row in subset],
            s=24,
            alpha=0.72,
            label=drone_type,
            color=colors.get(drone_type, "#666666"),
        )
    ax.set_title("Absolute Error vs BBox Area Ratio")
    ax.set_xlabel("bbox_area_ratio")
    ax.set_ylabel("Absolute error (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    plt.rcParams["figure.dpi"] = old_dpi
    return output_paths


def _format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return "{:.{digits}f}".format(float(value), digits=digits)


def _write_analysis(
    path: Path,
    overall: Dict[str, Any],
    by_drone: Sequence[Dict[str, Any]],
    by_distance: Sequence[Dict[str, Any]],
    by_drone_distance: Sequence[Dict[str, Any]],
    metadata_note: str,
) -> None:
    worst_distances = sorted(by_distance, key=lambda row: float(row["mae"]), reverse=True)[:5]
    best_distances = sorted(by_distance, key=lambda row: float(row["mae"]))[:5]
    worst_drone_distance = sorted(by_drone_distance, key=lambda row: float(row["mae"]), reverse=True)[:8]

    lines = [
        "# Attempt 5 Study 02 External Nenrus Analysis",
        "",
        "## External Verdict",
        "",
    ]
    if float(overall.get("overprediction_rate", 0.0)) >= 0.95:
        lines.extend(
            [
                "The exported Attempt 5 ensemble does **not** generalize well to the prepared Nenrus dataset as a ready-to-use model.",
                "",
                "The dominant failure mode is systematic overprediction:",
                "",
                "- overprediction rate: `{}`".format(_format_metric(overall["overprediction_rate"])),
                "- underprediction rate: `{}`".format(_format_metric(overall["underprediction_rate"])),
                "- mean signed error: `{}m`".format(_format_metric(overall["mean_signed_error"])),
                "",
                "This means the model is not just noisy on Nenrus; it is biased upward across the dataset.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The exported Attempt 5 ensemble shows mixed external-domain behavior.",
                "The detailed sections below should be used to identify where it works and where it fails.",
                "",
            ]
        )
    lines.extend(
        [
        "## Overall Result",
        "",
        "- Count: `{}`".format(overall["count"]),
        "- MAE: `{}m`".format(_format_metric(overall["mae"])),
        "- Median absolute error: `{}m`".format(_format_metric(overall["median_absolute_error"])),
        "- Mean relative error: `{}`".format(_format_metric(overall["mean_relative_error"])),
        "- RMSE: `{}m`".format(_format_metric(overall["rmse"])),
        "- R2: `{}`".format(_format_metric(overall["r2"])),
        "- Within 5m: `{}`".format(_format_metric(overall["within_5m_rate"])),
        "- Within 10m: `{}`".format(_format_metric(overall["within_10m_rate"])),
        "- Mean signed error: `{}m`".format(_format_metric(overall["mean_signed_error"])),
        "",
        "## Main Takeaway",
        "",
        "Attempt 4's held-out performance was strong because train/test came from the same original data distribution.",
        "Nenrus is a different visual domain with real-drone images, different bbox scale behavior, shorter distance ranges, and missing weather/time metadata.",
        "The external result therefore exposes a real domain shift.",
        "",
        "The model can still be useful diagnostically: it tells us that bbox geometry learned from the original dataset does not transfer cleanly to these real-drone images.",
        "But it should not be treated as a calibrated ready model for Nenrus-style images without an adaptation/calibration step.",
        "",
        "## Metadata Note",
        "",
        metadata_note,
        "",
        "## By Drone Type / Source",
        "",
        ]
    )
    for row in sorted(by_drone, key=lambda item: str(item["drone_type"])):
        lines.append(
            "- `{}`: count `{}`, MAE `{}m`, mean signed error `{}m`, within 10m `{}`".format(
                row["drone_type"],
                row["count"],
                _format_metric(row["mae"]),
                _format_metric(row["mean_signed_error"]),
                _format_metric(row["within_10m_rate"]),
            )
        )
    lines.extend(["", "## Best Distance Buckets", ""])
    for row in best_distances:
        lines.append(
            "- `{}`m: count `{}`, MAE `{}m`, mean signed error `{}m`".format(
                row["distance_m"],
                row["count"],
                _format_metric(row["mae"]),
                _format_metric(row["mean_signed_error"]),
            )
        )
    lines.extend(["", "## Worst Distance Buckets", ""])
    for row in worst_distances:
        lines.append(
            "- `{}`m: count `{}`, MAE `{}m`, mean signed error `{}m`".format(
                row["distance_m"],
                row["count"],
                _format_metric(row["mae"]),
                _format_metric(row["mean_signed_error"]),
            )
        )
    lines.extend(["", "## Worst Drone-Type x Distance Cells", ""])
    for row in worst_drone_distance:
        lines.append(
            "- `{}` `{}`m: count `{}`, MAE `{}m`, mean signed error `{}m`".format(
                row["drone_type"],
                row["distance_m"],
                row["count"],
                _format_metric(row["mae"]),
                _format_metric(row["mean_signed_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "This is an external-domain test, not a same-distribution held-out test.",
            "The model was trained on synthetic/controlled drone imagery with known weather and time metadata, while Nenrus contains real-drone imagery and does not provide those metadata fields.",
            "If the MAE is substantially higher than Attempt 4, that does not mean the previous model was invalid; it means the domain shift is real.",
            "",
        "The key question for the next step is whether the error pattern is systematic enough to correct without retraining on the external test set.",
        "Useful signs would be consistent bias by distance, consistent bias by drone type, or a predictable failure at near/far ranges.",
        "",
        "In this run the bias is systematic: predictions are consistently too high.",
        "That suggests the next useful test is not another model-family comparison, but a calibration/domain-adaptation study designed explicitly for external real-drone imagery.",
        "",
        ]
    )
    _write_text(path, "\n".join(lines))


def run_evaluation(config_path: Optional[Path] = None, recompute_features: bool = False) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _load_yaml(config_path)

    dataset_config = dict(config["dataset"])
    model_config = dict(config["model"])
    feature_config = dict(config["features"])
    depth_config = dict(config["depth_model"])
    plot_config = dict(config.get("plots", {}))

    manifest_csv = _resolve_repo_path(str(dataset_config["manifest_csv"]))
    ensemble_joblib = _resolve_repo_path(str(model_config["ensemble_joblib"]))
    output_root = _resolve_repo_path(str(config["output_root"]))
    features_dir = output_root / "features"
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    feature_table_csv = features_dir / "nenrus_feature_table.csv"
    manifest_rows = _read_csv(manifest_csv)
    raw_depth_feature_name = str(feature_config["raw_depth_feature_name"])

    if (
        bool(feature_config.get("reuse_feature_table", True))
        and feature_table_csv.exists()
        and not recompute_features
    ):
        print("[attempt5-study02] Reusing feature table: {}".format(feature_table_csv))
        feature_rows = _read_csv(feature_table_csv)
    else:
        print("[attempt5-study02] Loading depth model...")
        depth_model = load_depth_anything_v2(
            model_name=str(depth_config.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")),
            device=depth_config.get("device"),
            local_files_only=bool(depth_config.get("local_files_only", True)),
        )
        context_payload = dict(feature_config["context_window"])
        context_spec = ContextWindowSpec(
            name=str(context_payload["name"]),
            scale=float(context_payload["scale"]) if context_payload.get("scale") is not None else None,
        )
        feature_rows = _build_feature_rows(
            manifest_rows=manifest_rows,
            depth_model=depth_model,
            raw_depth_feature_name=raw_depth_feature_name,
            context_spec=context_spec,
            aggregation_method=str(feature_config["aggregation_method"]),
            missing_weather_value=str(dataset_config.get("missing_weather_value", "unknown")),
            missing_time_of_day_value=str(dataset_config.get("missing_time_of_day_value", "unknown")),
            progress_log_every=int(feature_config.get("progress_log_every", 20)),
        )
        _write_csv_rows(feature_rows, feature_table_csv)

    ensemble_metadata = joblib.load(ensemble_joblib)
    rf_model = joblib.load(str(ensemble_metadata["rf_model_path"]))
    xgb_model = joblib.load(str(ensemble_metadata["xgb_model_path"]))
    schema = dict(ensemble_metadata["schema"])
    blend_weights = dict(ensemble_metadata["blend_weights"])

    x = _build_tabular_matrix(
        feature_rows,
        list(schema["continuous_feature_names"]),
        list(schema["categorical_feature_names"]),
        dict(schema["categorical_levels"]),
    )
    rf_predictions = np.asarray(rf_model.predict(x), dtype=np.float64)
    xgb_predictions = np.asarray(xgb_model.predict(x), dtype=np.float64)
    ensemble_predictions = (
        float(blend_weights["rf_weight"]) * rf_predictions
        + float(blend_weights["xgb_weight"]) * xgb_predictions
    )

    predictions = _prediction_rows(
        feature_rows=feature_rows,
        rf_predictions=rf_predictions,
        xgb_predictions=xgb_predictions,
        ensemble_predictions=ensemble_predictions,
    )
    predictions_csv = reports_dir / "predictions.csv"
    _write_csv_rows(predictions, predictions_csv)

    overall = _metrics(predictions)
    by_distance = _group_metric_rows(predictions, ["distance_m"])
    by_drone = _group_metric_rows(predictions, ["drone_type"])
    by_drone_distance = _group_metric_rows(predictions, ["drone_type", "distance_m"])

    _write_json(overall, reports_dir / "overall_metrics.json")
    _write_csv_rows(by_distance, reports_dir / "by_distance.csv")
    _write_csv_rows(by_drone, reports_dir / "by_drone_type.csv")
    _write_csv_rows(by_drone_distance, reports_dir / "by_drone_type_and_distance.csv")

    plot_paths: List[str] = []
    if bool(plot_config.get("enabled", True)):
        plot_paths = _make_plots(
            predictions=predictions,
            by_distance=by_distance,
            by_drone=by_drone,
            by_drone_distance=by_drone_distance,
            plots_dir=plots_dir,
            dpi=int(plot_config.get("dpi", 160)),
        )

    metadata_note = (
        "Nenrus does not provide weather/time_of_day metadata. "
        "Both fields were set to `unknown`, which makes all trained metadata one-hot columns zero."
    )
    _write_analysis(
        path=reports_dir / "analysis.md",
        overall=overall,
        by_drone=by_drone,
        by_distance=by_distance,
        by_drone_distance=by_drone_distance,
        metadata_note=metadata_note,
    )

    summary = {
        "study_name": str(config.get("study_name", "study02_external_nenrus_evaluation")),
        "dataset_manifest_csv": str(manifest_csv),
        "ensemble_joblib": str(ensemble_joblib),
        "num_rows": len(predictions),
        "feature_table_csv": str(feature_table_csv),
        "predictions_csv": str(predictions_csv),
        "overall_metrics_json": str(reports_dir / "overall_metrics.json"),
        "by_distance_csv": str(reports_dir / "by_distance.csv"),
        "by_drone_type_csv": str(reports_dir / "by_drone_type.csv"),
        "by_drone_type_and_distance_csv": str(reports_dir / "by_drone_type_and_distance.csv"),
        "analysis_md": str(reports_dir / "analysis.md"),
        "plots": plot_paths,
        "metadata_note": metadata_note,
        "overall": overall,
    }
    _write_json(summary, reports_dir / "summary.json")
    print("[attempt5-study02] Wrote summary: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    run_evaluation(config_path=args.config, recompute_features=args.recompute_features)


if __name__ == "__main__":
    main()
