"""Evaluate the Attempt8 ensemble on the real Nenrus dataset without calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study03.run_study_03 import (  # noqa: E402
    _build_expected_feature_layout,
    _compute_noisy_feature_row,
    _resolve_jitter_specs,
)
from pipeline.data.loader import yolo_normalized_to_absolute_bbox  # noqa: E402
from pipeline.depth.depth_model import load_depth_anything_v2  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class RealAnnotation:
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class RealSample:
    image_path: Path
    label_path: Path
    true_distance_m: float
    weather: str
    time_of_day: str
    image_width: int
    image_height: int
    annotation: RealAnnotation


class RealImageLoader:
    @staticmethod
    def load_image(sample: RealSample) -> Image.Image:
        with Image.open(sample.image_path) as image:
            return image.convert("RGB").copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Attempt8 ensemble on real Nenrus data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Ignore cached real feature table and recompute depth/jitter features.",
    )
    parser.add_argument(
        "--skip-model-export",
        action="store_true",
        help="Use existing exported model artifacts if present.",
    )
    return parser.parse_args()


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML did not contain a mapping: {path}")
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON did not contain a mapping: {path}")
    return payload


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _round_float(value: float, digits: int = 10) -> float:
    return float(round(float(value), digits))


def _load_feature_names(selection_csv: Path, source_model_name: str) -> List[str]:
    rows = _read_csv_rows(selection_csv)
    matching = [row for row in rows if str(row.get("model_name", "")) == source_model_name]
    if not matching:
        raise ValueError(f"Could not find model '{source_model_name}' in {selection_csv}")
    raw_value = str(matching[0]["feature_names"])
    feature_names = [item.strip() for item in raw_value.split("|") if item.strip()]
    if not feature_names:
        raise ValueError(f"Feature selection for '{source_model_name}' was empty.")
    return feature_names


def _candidate_by_name(candidates: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for candidate in candidates:
        payload = dict(candidate)
        candidate_name = str(payload.pop("name"))
        if candidate_name == name:
            return payload
    raise ValueError(f"Candidate '{name}' was not found.")


def _fit_rf(params: Dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params.get("max_depth") is None else int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params.get("max_features", "sqrt"),
        random_state=int(params["random_state"]),
        n_jobs=int(params.get("n_jobs", 1)),
    )


def _fit_xgb(params: Dict[str, Any]) -> Any:
    if XGBRegressor is None:
        raise RuntimeError(f"xgboost could not be imported: {XGB_IMPORT_ERROR}")
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        random_state=int(params["random_state"]),
        n_jobs=int(params.get("n_jobs", 1)),
        tree_method=str(params.get("tree_method", "hist")),
        verbosity=int(params.get("verbosity", 0)),
    )


def _matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def _target_values(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)


def _add_one_hot_features(row: Dict[str, Any], feature_names: Sequence[str]) -> None:
    for feature_name in feature_names:
        if "=" not in feature_name:
            continue
        base_name, level = feature_name.split("=", 1)
        row[feature_name] = 1.0 if str(row.get(base_name, "")) == level else 0.0


def _prepare_training_rows(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    numeric_feature_names = [name for name in feature_names if "=" not in name]
    for row in rows:
        payload = dict(row)
        payload["true_distance_m"] = float(payload["true_distance_m"])
        for feature_name in numeric_feature_names:
            payload[feature_name] = float(payload[feature_name])
        _add_one_hot_features(payload, feature_names)
        prepared.append(payload)
    missing = [name for name in feature_names if name not in prepared[0]]
    if missing:
        raise ValueError(f"Training rows are missing required model features: {missing}")
    return prepared


def export_attempt8_ensemble(config: Dict[str, Any], output_root: Path, skip_if_present: bool) -> Dict[str, Any]:
    sources = dict(config["sources"])
    study_all_config = _read_yaml(_resolve_repo_path(sources["study_all_config_yaml"]))
    study_all_summary = _read_json(_resolve_repo_path(sources["study_all_summary_json"]))
    feature_table_csv = _resolve_repo_path(sources["study_all_aggregated_feature_table_csv"])
    selection_csv = _resolve_repo_path(sources["aggregated_feature_selection_csv"])

    model_cfg = dict(config["model"])
    source_model_name = str(model_cfg["source_model_name"])
    feature_names = _load_feature_names(selection_csv, source_model_name)
    best_rf_name = str(study_all_summary["best_aggregated_rf_candidate"])
    best_xgb_name = str(study_all_summary["best_aggregated_xgb_candidate"])
    best_blend = dict(study_all_summary["best_blend_row"])
    rf_weight = float(best_blend["rf_weight"])
    xgb_weight = float(best_blend["xgb_weight"])

    model_dir = output_root / "models" / str(model_cfg.get("export_scope", "full_study_all_dataset"))
    rf_path = model_dir / "rf_model.joblib"
    xgb_path = model_dir / "xgb_model.joblib"
    ensemble_path = model_dir / "ensemble.joblib"
    schema_path = output_root / "reports" / "model_schema.json"
    summary_path = output_root / "reports" / "model_export_summary.json"

    if skip_if_present and ensemble_path.exists() and rf_path.exists() and xgb_path.exists():
        print(f"[attempt8-study01] Reusing exported ensemble: {ensemble_path}")
        metadata = joblib.load(ensemble_path)
        schema = dict(metadata["schema"])
        return {
            "ensemble_joblib": str(ensemble_path),
            "rf_model_joblib": str(rf_path),
            "xgb_model_joblib": str(xgb_path),
            "model_schema_json": str(schema_path),
            "train_rows": int(schema.get("train_rows", 0)),
            "feature_count": len(list(schema.get("feature_names", []))),
            "rf_candidate_name": str(schema.get("rf_candidate_name", "")),
            "xgb_candidate_name": str(schema.get("xgb_candidate_name", "")),
            "blend_weights": dict(schema.get("blend_weights", {})),
        }

    aggregated_cfg = dict(study_all_config["aggregated_models"])
    rf_params = _candidate_by_name(list(aggregated_cfg["rf_candidates"]), best_rf_name)
    xgb_params = _candidate_by_name(list(aggregated_cfg["xgb_candidates"]), best_xgb_name)
    train_rows = _prepare_training_rows(_read_csv_rows(feature_table_csv), feature_names)

    print(
        "[attempt8-study01] Fitting full StudyAll ensemble | rows={} | RF={} | XGB={} | blend={:.2f}/{:.2f}".format(
            len(train_rows),
            best_rf_name,
            best_xgb_name,
            rf_weight,
            xgb_weight,
        )
    )
    rf_model = _fit_rf(rf_params)
    rf_model.fit(_matrix(train_rows, feature_names), _target_values(train_rows))
    xgb_model = _fit_xgb(xgb_params)
    xgb_model.fit(_matrix(train_rows, feature_names), _target_values(train_rows))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    schema = {
        "model_name": "attempt8_aggregated_rf_xgb_ensemble",
        "source_model_name": source_model_name,
        "feature_names": list(feature_names),
        "rf_candidate_name": best_rf_name,
        "xgb_candidate_name": best_xgb_name,
        "rf_params": dict(rf_params),
        "xgb_params": dict(xgb_params),
        "blend_weights": {"rf_weight": rf_weight, "xgb_weight": xgb_weight},
        "clip_min_distance_m": float(model_cfg.get("clip_min_distance_m", 0.0)),
        "train_rows": len(train_rows),
        "training_feature_table_csv": str(feature_table_csv),
    }
    joblib.dump(
        {
            "model_family": "attempt8_aggregated_rf_xgb_weighted_average",
            "scope": str(model_cfg.get("export_scope", "full_study_all_dataset")),
            "rf_model_path": str(rf_path),
            "xgb_model_path": str(xgb_path),
            "schema": schema,
            "blend_weights": dict(schema["blend_weights"]),
            "clip_min_distance_m": float(schema["clip_min_distance_m"]),
            "load_note": "Load RF/XGB paths, build schema feature_names, then blend predictions.",
        },
        ensemble_path,
    )
    _write_json(schema, schema_path)
    summary = {
        "ensemble_joblib": str(ensemble_path),
        "rf_model_joblib": str(rf_path),
        "xgb_model_joblib": str(xgb_path),
        "model_schema_json": str(schema_path),
        "train_rows": len(train_rows),
        "feature_count": len(feature_names),
        "rf_candidate_name": best_rf_name,
        "xgb_candidate_name": best_xgb_name,
        "blend_weights": dict(schema["blend_weights"]),
    }
    _write_json(summary, summary_path)
    return summary


def _parse_yolo_label(label_path: Path, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    boxes: List[Tuple[int, int, int, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = parts
        boxes.append(
            yolo_normalized_to_absolute_bbox(
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
                image_width=image_width,
                image_height=image_height,
            )
        )
    if not boxes:
        raise ValueError(f"No usable YOLO bbox found in {label_path}")
    return max(boxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _manifest_row_to_sample(row: Dict[str, Any], missing_weather: str, missing_time: str) -> RealSample:
    image_path = Path(str(row["image_path"]))
    label_path = Path(str(row["label_path"]))
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    bbox = _parse_yolo_label(label_path, image_width=image_width, image_height=image_height)
    return RealSample(
        image_path=image_path,
        label_path=label_path,
        true_distance_m=float(row.get("distance_m", row.get("distance", 0.0))),
        weather=missing_weather,
        time_of_day=missing_time,
        image_width=int(image_width),
        image_height=int(image_height),
        annotation=RealAnnotation(bbox=bbox),
    )


def build_or_load_real_features(
    config: Dict[str, Any],
    output_root: Path,
    feature_names: Sequence[str],
    recompute_features: bool,
) -> List[Dict[str, Any]]:
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config["features"])
    sources = dict(config["sources"])
    manifest_csv = _resolve_repo_path(dataset_cfg["manifest_csv"])
    feature_table_csv = output_root / "features" / "nenrus_attempt8_jitter_feature_table.csv"

    if (
        bool(feature_cfg.get("reuse_feature_table", True))
        and feature_table_csv.exists()
        and not recompute_features
    ):
        print(f"[attempt8-study01] Reusing real feature table: {feature_table_csv}")
        rows = _read_csv_rows(feature_table_csv)
        for row in rows:
            _add_one_hot_features(row, feature_names)
        return rows

    jitter_config = _read_yaml(_resolve_repo_path(sources["jitter_feature_config_yaml"]))
    study_config = dict(jitter_config["study"])
    feature_layout = _build_expected_feature_layout(study_config)
    jitter_specs = _resolve_jitter_specs(study_config)
    depth_cfg = dict(config["depth_model"])

    print("[attempt8-study01] Loading depth model for real jitter features...")
    depth_model = load_depth_anything_v2(
        model_name=str(depth_cfg.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")),
        device=depth_cfg.get("device"),
        local_files_only=bool(depth_cfg.get("local_files_only", True)),
    )
    loader = RealImageLoader()
    manifest_rows = _read_csv_rows(manifest_csv)
    output_rows: List[Dict[str, Any]] = []
    progress_every = int(feature_cfg.get("progress_log_every", 20))
    missing_weather = str(dataset_cfg.get("missing_weather_value", "unknown"))
    missing_time = str(dataset_cfg.get("missing_time_of_day_value", "unknown"))

    for index, manifest_row in enumerate(manifest_rows, start=1):
        sample = _manifest_row_to_sample(manifest_row, missing_weather, missing_time)
        feature_row = _compute_noisy_feature_row(
            sample=sample,
            loader=loader,
            depth_model=depth_model,
            context_specs=feature_layout["context_specs"],
            aggregation_methods=feature_layout["aggregation_methods"],
            jitter_specs=jitter_specs,
        )
        drone_type = str(manifest_row.get("drone_type", "unknown"))
        feature_row.update(
            {
                "sample_id": f"{drone_type}::{sample.image_path.name}",
                "drone_type": drone_type,
                "distance_m": _round_float(sample.true_distance_m),
                "source_group_id": str(sample.image_path),
            }
        )
        _add_one_hot_features(feature_row, feature_names)
        output_rows.append(feature_row)
        if index == 1 or index == len(manifest_rows) or index % progress_every == 0:
            print(
                "[attempt8-study01] Real features {}/{} | {} | distance={}m".format(
                    index,
                    len(manifest_rows),
                    drone_type,
                    sample.true_distance_m,
                )
            )

    _write_csv_rows(output_rows, feature_table_csv)
    feature_summary = {
        "feature_table_csv": str(feature_table_csv),
        "num_rows": len(output_rows),
        "num_jitter_boxes": len(jitter_specs),
        "contexts": [spec.name for spec in feature_layout["context_specs"]],
        "aggregation_methods": list(feature_layout["aggregation_methods"]),
        "model_feature_count": len(feature_names),
        "missing_weather_value": missing_weather,
        "missing_time_of_day_value": missing_time,
    }
    _write_json(feature_summary, output_root / "features" / "summary.json")
    return output_rows


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    errors = np.asarray([float(row["signed_error_m"]) for row in rows], dtype=np.float64)
    abs_errors = np.abs(errors)
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    pred_values = np.asarray([float(row["predicted_distance_m"]) for row in rows], dtype=np.float64)
    rel_errors = np.divide(abs_errors, true_values, out=np.zeros_like(abs_errors), where=true_values > 0)
    sse = float(np.sum((pred_values - true_values) ** 2))
    centered = true_values - float(np.mean(true_values))
    sst = float(np.sum(centered**2))
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


def _group_metric_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output: List[Dict[str, Any]] = []
    def sort_key(item: Tuple[Tuple[str, ...], List[Dict[str, Any]]]) -> Tuple[Any, ...]:
        values = item[0]
        output: List[Any] = []
        for key, value in zip(keys, values):
            if key == "distance_m":
                output.append(float(value))
            else:
                output.append(value)
        return tuple(output)

    for key_values, group_rows in sorted(grouped.items(), key=sort_key):
        payload = {key: value for key, value in zip(keys, key_values)}
        payload.update(_metrics(group_rows))
        output.append(payload)
    return output


def _prediction_rows(
    feature_rows: Sequence[Dict[str, Any]],
    rf_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
    ensemble_predictions: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row, rf_pred, xgb_pred, pred in zip(feature_rows, rf_predictions, xgb_predictions, ensemble_predictions):
        true_distance = float(row["true_distance_m"])
        signed_error = float(pred) - true_distance
        rows.append(
            {
                "model_name": "attempt8_ensemble_uncalibrated",
                "split_name": "real_external_uncalibrated",
                "sample_id": str(row["sample_id"]),
                "drone_type": str(row["drone_type"]),
                "distance_m": float(row["distance_m"]),
                "true_distance_m": true_distance,
                "predicted_distance_m": float(pred),
                "rf_prediction_m": float(rf_pred),
                "xgb_prediction_m": float(xgb_pred),
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
                "image_path": str(row["image_path"]),
                "label_path": str(row["label_path"]),
                "weather": str(row["weather"]),
                "time_of_day": str(row["time_of_day"]),
                "bbox_width_norm__jitter_median": float(row["bbox_width_norm__jitter_median"]),
                "bbox_area_ratio__jitter_median": float(row["bbox_area_ratio__jitter_median"]),
                "raw_depth_jitter_median": float(row["bbox_only__inner50_median__object_depth__jitter_median"]),
            }
        )
    return rows


def _make_plots(
    predictions: Sequence[Dict[str, Any]],
    by_distance: Sequence[Dict[str, Any]],
    by_drone: Sequence[Dict[str, Any]],
    output_root: Path,
    dpi: int,
) -> List[str]:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    old_dpi = plt.rcParams.get("figure.dpi", 100)
    plt.rcParams["figure.dpi"] = dpi
    output_paths: List[str] = []
    drone_types = sorted({str(row["drone_type"]) for row in predictions})
    colors = {"Kongsberg": "#4e79a7", "Vestfold": "#f28e2b", "unknown": "#666666"}

    path = plots_dir / "predicted_vs_true_by_drone_type.png"
    fig, ax = plt.subplots(figsize=(8, 6))
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
    values = [float(row["true_distance_m"]) for row in predictions] + [
        float(row["predicted_distance_m"]) for row in predictions
    ]
    ax.plot([min(values), max(values)], [min(values), max(values)], color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("Raw Attempt8 Ensemble on Real Drone Images")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Predicted distance (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "mae_by_drone_type.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [str(row["drone_type"]) for row in by_drone]
    values = [float(row["mae"]) for row in by_drone]
    ax.bar(labels, values, color=[colors.get(label, "#666666") for label in labels])
    ax.set_title("Raw MAE by Real Dataset")
    ax.set_ylabel("MAE (m)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    output_paths.append(str(path))

    path = plots_dir / "mae_by_distance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_rows = sorted(by_distance, key=lambda row: float(row["distance_m"]))
    ax.plot(
        [float(row["distance_m"]) for row in sorted_rows],
        [float(row["mae"]) for row in sorted_rows],
        marker="o",
        linewidth=2.0,
        color="#0f766e",
    )
    ax.set_title("Raw MAE by True Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("MAE (m)")
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
    return f"{float(value):.{digits}f}"


def _write_analysis(
    path: Path,
    overall: Dict[str, Any],
    by_drone: Sequence[Dict[str, Any]],
    by_distance: Sequence[Dict[str, Any]],
    model_export: Dict[str, Any],
) -> None:
    lines = [
        "# Study 01: Raw Attempt8 Ensemble on Real Nenrus Data",
        "",
        "## Protocol",
        "",
        "- The Attempt8 ensemble was exported from `study_all` and trained on all synthetic StudyAll rows.",
        "- The real Nenrus rows were used only for external evaluation.",
        "- No calibration, fine-tuning, or real-domain retraining was applied.",
        "",
        "## Overall Raw Result",
        "",
        f"- Count: `{overall['count']}`",
        f"- MAE: `{_format_metric(overall['mae'])}m`",
        f"- RMSE: `{_format_metric(overall['rmse'])}m`",
        f"- R2: `{_format_metric(overall['r2'])}`",
        f"- Mean signed error: `{_format_metric(overall['mean_signed_error'])}m`",
        f"- Within 10m: `{_format_metric(overall['within_10m_rate'])}`",
        f"- Overprediction rate: `{_format_metric(overall['overprediction_rate'])}`",
        "",
        "## By Real Dataset / Drone Type",
        "",
    ]
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
    lines.extend(["", "## By Distance", ""])
    for row in sorted(by_distance, key=lambda item: float(item["distance_m"])):
        lines.append(
            "- `{}`m: count `{}`, MAE `{}m`, mean signed error `{}m`".format(
                row["distance_m"],
                row["count"],
                _format_metric(row["mae"]),
                _format_metric(row["mean_signed_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Exported Model",
            "",
            f"- Ensemble: `{model_export['ensemble_joblib']}`",
            f"- Training rows: `{model_export.get('train_rows', 'n/a')}`",
            f"- RF candidate: `{model_export.get('rf_candidate_name', 'n/a')}`",
            f"- XGB candidate: `{model_export.get('xgb_candidate_name', 'n/a')}`",
            f"- Blend weights: `{model_export.get('blend_weights', {})}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_study(config_path: Optional[Path] = None, recompute_features: bool = False, skip_model_export: bool = False) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _read_yaml(config_path)
    output_root = _resolve_repo_path(config["output_root"])
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_export = export_attempt8_ensemble(config, output_root, skip_if_present=skip_model_export)
    ensemble_metadata = joblib.load(model_export["ensemble_joblib"])
    schema = dict(ensemble_metadata["schema"])
    feature_names = list(schema["feature_names"])

    feature_rows = build_or_load_real_features(
        config=config,
        output_root=output_root,
        feature_names=feature_names,
        recompute_features=recompute_features,
    )
    rf_model = joblib.load(str(ensemble_metadata["rf_model_path"]))
    xgb_model = joblib.load(str(ensemble_metadata["xgb_model_path"]))
    rf_predictions = np.asarray(rf_model.predict(_matrix(feature_rows, feature_names)), dtype=np.float64)
    xgb_predictions = np.asarray(xgb_model.predict(_matrix(feature_rows, feature_names)), dtype=np.float64)
    blend_weights = dict(ensemble_metadata["blend_weights"])
    ensemble_predictions = (
        float(blend_weights["rf_weight"]) * rf_predictions
        + float(blend_weights["xgb_weight"]) * xgb_predictions
    )
    ensemble_predictions = np.maximum(
        ensemble_predictions,
        float(ensemble_metadata.get("clip_min_distance_m", 0.0)),
    )

    predictions = _prediction_rows(feature_rows, rf_predictions, xgb_predictions, ensemble_predictions)
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
    if bool(dict(config.get("plots", {})).get("enabled", True)):
        plot_paths = _make_plots(
            predictions=predictions,
            by_distance=by_distance,
            by_drone=by_drone,
            output_root=output_root,
            dpi=int(dict(config.get("plots", {})).get("dpi", 160)),
        )

    analysis_md = reports_dir / "analysis.md"
    _write_analysis(analysis_md, overall, by_drone, by_distance, model_export)

    summary = {
        "study_name": str(config.get("study_name", "study01_real_nenrus_uncalibrated_attempt8_ensemble")),
        "config_path": str(config_path),
        "dataset_manifest_csv": str(_resolve_repo_path(dict(config["dataset"])["manifest_csv"])),
        "num_rows": len(predictions),
        "model_export": model_export,
        "feature_table_csv": str(output_root / "features" / "nenrus_attempt8_jitter_feature_table.csv"),
        "predictions_csv": str(predictions_csv),
        "overall_metrics_json": str(reports_dir / "overall_metrics.json"),
        "by_distance_csv": str(reports_dir / "by_distance.csv"),
        "by_drone_type_csv": str(reports_dir / "by_drone_type.csv"),
        "by_drone_type_and_distance_csv": str(reports_dir / "by_drone_type_and_distance.csv"),
        "analysis_md": str(analysis_md),
        "plots": plot_paths,
        "overall": overall,
    }
    _write_json(summary, reports_dir / "summary.json")
    print(f"[attempt8-study01] Wrote summary: {reports_dir / 'summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    run_study(
        config_path=args.config,
        recompute_features=bool(args.recompute_features),
        skip_model_export=bool(args.skip_model_export),
    )


if __name__ == "__main__":
    main()
