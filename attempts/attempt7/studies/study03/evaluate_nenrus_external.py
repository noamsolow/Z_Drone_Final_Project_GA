"""
Evaluate the exported Attempt 7 aggregated-jitter ensemble on the prepared Nenrus dataset.
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

from attempts.attempt3.studies.study01.run_study_01 import _write_csv_rows, _write_json
from attempts.attempt3.studies.study03.run_study_03 import (
    _aggregate_numeric_values,
    _build_geometry_feature_dict,
    _jitter_bbox,
    _round_float,
)
from pipeline.data.loader import yolo_normalized_to_absolute_bbox
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import ContextWindowSpec, compute_depth_representation, extract_context_crop


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported attempt7 ensemble on Nenrus.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--recompute-features", action="store_true")
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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    selected_feature_names: Sequence[str],
    jitter_specs: Sequence[Dict[str, Any]],
    missing_weather_value: str,
    missing_time_of_day_value: str,
    progress_log_every: int,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    need_raw_depth = "bbox_only__inner50_median__object_depth__jitter_median" in selected_feature_names

    for index, manifest_row in enumerate(manifest_rows, start=1):
        image_path = Path(str(manifest_row["image_path"]))
        label_path = Path(str(manifest_row["label_path"]))
        drone_type = _drone_type_from_manifest_row(manifest_row)
        true_distance = _distance_from_manifest_row(manifest_row)

        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        image_width, image_height = image.size
        bbox = _parse_yolo_label(label_path, image_width=image_width, image_height=image_height)

        jittered_bboxes = [
            _jitter_bbox(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                jitter_spec=jitter_spec,
            )
            for jitter_spec in jitter_specs
        ]

        row: Dict[str, Any] = {
            "sample_id": "{}::{}".format(drone_type, image_path.name),
            "drone_type": drone_type,
            "distance_m": _round_float(true_distance),
            "true_distance_m": _round_float(true_distance),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "weather": missing_weather_value,
            "time_of_day": missing_time_of_day_value,
        }

        for feature_name in [
            "bbox_width_px",
            "bbox_height_px",
            "bbox_width_norm",
            "bbox_height_norm",
            "bbox_area_ratio",
            "bbox_aspect_ratio",
            "bbox_center_x_norm",
            "bbox_center_y_norm",
        ]:
            values = [
                _build_geometry_feature_dict(
                    bbox=jittered_bbox,
                    image_width=image_width,
                    image_height=image_height,
                )[feature_name]
                for jittered_bbox in jittered_bboxes
            ]
            feature_median, feature_std = _aggregate_numeric_values(values)
            row["{}__jitter_median".format(feature_name)] = _round_float(feature_median)
            row["{}__jitter_std".format(feature_name)] = _round_float(feature_std)

        if need_raw_depth:
            depth_values: List[float] = []
            for jittered_bbox in jittered_bboxes:
                crop_data = extract_context_crop(
                    image=image,
                    bbox=jittered_bbox,
                    context_spec=ContextWindowSpec(name="bbox_only", scale=1.0),
                )
                depth_map = estimate_relative_depth(crop_data["image"], depth_model)
                representation = compute_depth_representation(
                    depth_map=depth_map,
                    bbox=crop_data["local_bbox"],
                    aggregation_method="inner50_median",
                )
                depth_values.append(float(representation["object_depth"]))
            depth_median, _ = _aggregate_numeric_values(depth_values)
            row["bbox_only__inner50_median__object_depth__jitter_median"] = _round_float(depth_median)

        for feature_name in selected_feature_names:
            if "=" in feature_name:
                base_name, level = feature_name.split("=", 1)
                row[feature_name] = 1.0 if str(row.get(base_name, "")) == level else 0.0

        output_rows.append(row)
        if index == 1 or index == len(manifest_rows) or index % progress_log_every == 0:
            print(
                "[attempt7-study03] Features {}/{} | {} | distance={}m".format(
                    index,
                    len(manifest_rows),
                    drone_type,
                    true_distance,
                )
            )
    return output_rows


def _matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    errors = np.asarray([float(row["signed_error_m"]) for row in rows], dtype=np.float64)
    abs_errors = np.abs(errors)
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    pred_values = np.asarray([float(row["predicted_distance_m"]) for row in rows], dtype=np.float64)
    rel_errors = np.divide(abs_errors, true_values, out=np.zeros_like(abs_errors), where=true_values > 0)
    if len(rows) == 0:
        return {"count": 0, "mae": None, "mean_relative_error": None}
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
                "bbox_area_ratio_jitter_median": float(row.get("bbox_area_ratio__jitter_median", 0.0)),
                "raw_depth_feature": float(row.get("bbox_only__inner50_median__object_depth__jitter_median", 0.0)),
                "rf_xgb_prediction_delta_m": float(rf_pred - xgb_pred),
            }
        )

    return output


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


def _make_basic_plots(
    predictions: Sequence[Dict[str, Any]],
    by_distance: Sequence[Dict[str, Any]],
    plots_dir: Path,
    dpi: int,
) -> List[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    old_dpi = plt.rcParams.get("figure.dpi", 100)
    plt.rcParams["figure.dpi"] = dpi
    paths: List[str] = []

    path = plots_dir / "predicted_vs_true.png"
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        [float(row["true_distance_m"]) for row in predictions],
        [float(row["predicted_distance_m"]) for row in predictions],
        s=22,
        alpha=0.72,
        color="#315f8c",
    )
    true_values = np.asarray([float(row["true_distance_m"]) for row in predictions], dtype=np.float64)
    pred_values = np.asarray([float(row["predicted_distance_m"]) for row in predictions], dtype=np.float64)
    min_axis = min(float(np.min(true_values)), float(np.min(pred_values)))
    max_axis = max(float(np.max(true_values)), float(np.max(pred_values)))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("Attempt 7 Predicted vs True Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Predicted distance (m)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    distance_rows = sorted(by_distance, key=lambda row: float(row["distance_m"]))
    path = plots_dir / "mae_by_distance.png"
    _save_bar_plot(
        [str(int(float(row["distance_m"]))) for row in distance_rows],
        [float(row["mae"]) for row in distance_rows],
        "Attempt 7 MAE by Distance",
        "MAE (m)",
        path,
        color="#365f46",
    )
    paths.append(str(path))
    plt.rcParams["figure.dpi"] = old_dpi
    return paths


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

    manifest_rows = _read_csv(manifest_csv)
    ensemble_metadata = joblib.load(ensemble_joblib)
    schema = dict(ensemble_metadata["schema"])
    selected_feature_names = list(schema["continuous_feature_names"])

    feature_table_csv = features_dir / "nenrus_feature_table.csv"
    if bool(feature_config.get("reuse_feature_table", True)) and feature_table_csv.exists() and not recompute_features:
        print("[attempt7-study03] Reusing feature table: {}".format(feature_table_csv))
        feature_rows = _read_csv(feature_table_csv)
    else:
        print("[attempt7-study03] Loading depth model...")
        depth_model = load_depth_anything_v2(
            model_name=str(depth_config.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")),
            device=depth_config.get("device"),
            local_files_only=bool(depth_config.get("local_files_only", True)),
        )
        feature_rows = _build_feature_rows(
            manifest_rows=manifest_rows,
            depth_model=depth_model,
            selected_feature_names=selected_feature_names,
            jitter_specs=[dict(row) for row in list(feature_config["jitter_boxes"])],
            missing_weather_value=str(dataset_config.get("missing_weather_value", "unknown")),
            missing_time_of_day_value=str(dataset_config.get("missing_time_of_day_value", "unknown")),
            progress_log_every=int(feature_config.get("progress_log_every", 20)),
        )
        _write_csv_rows(feature_rows, feature_table_csv)

    rf_model = joblib.load(str(ensemble_metadata["rf_model_path"]))
    xgb_model = joblib.load(str(ensemble_metadata["xgb_model_path"]))
    blend_weights = dict(ensemble_metadata["blend_weights"])

    x = _matrix(feature_rows, selected_feature_names)
    rf_predictions = np.asarray(rf_model.predict(x), dtype=np.float64)
    xgb_predictions = np.asarray(xgb_model.predict(x), dtype=np.float64)
    ensemble_predictions = (
        float(blend_weights["rf_weight"]) * rf_predictions
        + float(blend_weights["xgb_weight"]) * xgb_predictions
    )

    predictions = _prediction_rows(feature_rows, rf_predictions, xgb_predictions, ensemble_predictions)
    _write_csv_rows(predictions, reports_dir / "predictions.csv")

    overall = _metrics(predictions)
    by_distance = _group_metric_rows(predictions, ["distance_m"])
    by_drone = _group_metric_rows(predictions, ["drone_type"])
    _write_json(overall, reports_dir / "overall_metrics.json")
    _write_csv_rows(by_distance, reports_dir / "by_distance.csv")
    _write_csv_rows(by_drone, reports_dir / "by_drone_type.csv")

    plot_paths: List[str] = []
    if bool(plot_config.get("enabled", True)):
        plot_paths = _make_basic_plots(
            predictions=predictions,
            by_distance=by_distance,
            plots_dir=plots_dir,
            dpi=int(plot_config.get("dpi", 160)),
        )

    old_summary = _read_json(_resolve_repo_path(str(model_config["old_attempt5_summary_json"])))
    old_mae = float(dict(old_summary["overall"])["mae"])
    analysis = "\n".join(
        [
            "# Attempt 7 Study 03 External Nenrus Analysis",
            "",
            "- Attempt 7 MAE: `{:.4f}m`".format(float(overall["mae"])),
            "- Attempt 7 mean relative error: `{:.4f}`".format(float(overall["mean_relative_error"])),
            "- Old Attempt 5 ensemble MAE: `{:.4f}m`".format(old_mae),
            "- Delta vs old ensemble: `{:+.4f}m`".format(float(overall["mae"]) - old_mae),
            "",
            "Nenrus metadata are still missing, so weather/time one-hot features remain zero here too.",
        ]
    )
    _write_text(reports_dir / "analysis.md", analysis)

    summary = {
        "study_name": str(config["study_name"]),
        "dataset_manifest_csv": str(manifest_csv),
        "ensemble_joblib": str(ensemble_joblib),
        "num_rows": len(predictions),
        "feature_table_csv": str(feature_table_csv),
        "overall": overall,
        "old_attempt5_external_mae": old_mae,
        "delta_vs_old_attempt5_external_mae": float(overall["mae"]) - old_mae,
        "predictions_csv": str(reports_dir / "predictions.csv"),
        "overall_metrics_json": str(reports_dir / "overall_metrics.json"),
        "by_distance_csv": str(reports_dir / "by_distance.csv"),
        "by_drone_type_csv": str(reports_dir / "by_drone_type.csv"),
        "analysis_md": str(reports_dir / "analysis.md"),
        "plots": plot_paths,
    }
    _write_json(summary, reports_dir / "summary.json")
    print("[attempt7-study03] Wrote summary: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    run_evaluation(config_path=args.config, recompute_features=args.recompute_features)


if __name__ == "__main__":
    main()
