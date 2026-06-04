"""
Attempt 6 Study 01: train only on synthetic data while regularizing geometry,
then evaluate externally on real Nenrus data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
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

from pipeline.calibration.evaluation import evaluate_predictions
from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
COMPACT_GEOMETRY_FEATURE_NAMES = [
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt6 Study 01 geometry regularization.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config did not contain an object: {}".format(path))
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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


def _select_rows(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [rows[int(index)] for index in indices]


def _target_values(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)


def _depth_feature_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    names = sorted(
        name
        for name in rows[0].keys()
        if name.endswith("__object_depth")
    )
    if not names:
        raise ValueError("No depth feature columns ending with __object_depth were found.")
    return names


def _categorical_levels(
    rows: Sequence[Dict[str, Any]],
    categorical_feature_names: Sequence[str],
) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for feature_name in categorical_feature_names:
        output[feature_name] = sorted({str(row[feature_name]) for row in rows})
    return output


def _feature_names(
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Dict[str, Sequence[str]],
) -> List[str]:
    names = list(continuous_feature_names)
    for feature_name in categorical_feature_names:
        for level in categorical_levels.get(feature_name, []):
            names.append("{}={}".format(feature_name, level))
    return names


def _matrix(
    rows: Sequence[Dict[str, Any]],
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Dict[str, Sequence[str]],
) -> np.ndarray:
    matrix_rows: List[List[float]] = []
    for row in rows:
        values = [float(row[name]) for name in continuous_feature_names]
        for feature_name in categorical_feature_names:
            row_value = str(row.get(feature_name, ""))
            for level in categorical_levels.get(feature_name, []):
                values.append(1.0 if row_value == level else 0.0)
        matrix_rows.append(values)
    return np.asarray(matrix_rows, dtype=np.float64)


def _variant_feature_names(
    variant: Dict[str, Any],
    raw_depth_feature_name: str,
    all_depth_feature_names: Sequence[str],
) -> List[str]:
    depth_mode = str(variant["depth_mode"])
    geometry_mode = str(variant["geometry_mode"])

    if depth_mode == "raw":
        depth_names = [raw_depth_feature_name]
    elif depth_mode == "multiscale":
        depth_names = list(all_depth_feature_names)
    else:
        raise ValueError("Unsupported depth_mode '{}'.".format(depth_mode))

    if geometry_mode == "full":
        geometry_names = list(GEOMETRY_FEATURE_NAMES)
    elif geometry_mode == "compact":
        geometry_names = list(COMPACT_GEOMETRY_FEATURE_NAMES)
    elif geometry_mode == "none":
        geometry_names = []
    else:
        raise ValueError("Unsupported geometry_mode '{}'.".format(geometry_mode))

    return depth_names + geometry_names


def _geometry_medians(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    medians: Dict[str, float] = {}
    for name in GEOMETRY_FEATURE_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        medians[name] = float(np.median(values))
    return medians


def _corrupt_training_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    geometry_feature_names: Sequence[str],
    dropout_probability: float,
    noise_scale_std: float,
    medians: Dict[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)

        if noise_scale_std > 0.0:
            width_scale = float(np.clip(rng.normal(1.0, noise_scale_std), 0.35, 2.5))
            height_scale = float(np.clip(rng.normal(1.0, noise_scale_std), 0.35, 2.5))
            if "bbox_width_px" in geometry_feature_names:
                new_row["bbox_width_px"] = float(new_row["bbox_width_px"]) * width_scale
            if "bbox_width_norm" in geometry_feature_names:
                new_row["bbox_width_norm"] = float(new_row["bbox_width_norm"]) * width_scale
            if "bbox_height_px" in geometry_feature_names:
                new_row["bbox_height_px"] = float(new_row["bbox_height_px"]) * height_scale
            if "bbox_height_norm" in geometry_feature_names:
                new_row["bbox_height_norm"] = float(new_row["bbox_height_norm"]) * height_scale
            if "bbox_area_ratio" in geometry_feature_names:
                new_row["bbox_area_ratio"] = float(new_row["bbox_area_ratio"]) * width_scale * height_scale
            if "bbox_aspect_ratio" in geometry_feature_names:
                new_row["bbox_aspect_ratio"] = float(new_row["bbox_aspect_ratio"]) * width_scale / max(height_scale, 1e-6)
            if "bbox_center_x_norm" in geometry_feature_names:
                new_row["bbox_center_x_norm"] = float(np.clip(float(new_row["bbox_center_x_norm"]) + rng.normal(0.0, 0.025), 0.0, 1.0))
            if "bbox_center_y_norm" in geometry_feature_names:
                new_row["bbox_center_y_norm"] = float(np.clip(float(new_row["bbox_center_y_norm"]) + rng.normal(0.0, 0.025), 0.0, 1.0))

        if dropout_probability > 0.0:
            for feature_name in geometry_feature_names:
                if feature_name in GEOMETRY_FEATURE_NAMES and rng.random() < dropout_probability:
                    new_row[feature_name] = medians[feature_name]

        output.append(new_row)
    return output


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
        raise RuntimeError("xgboost could not be imported: {}".format(XGB_IMPORT_ERROR))
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


def _prediction_rows(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    *,
    variant_name: str,
    model_name: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        pred = float(prediction)
        abs_error = abs(pred - true_distance)
        output.append(
            {
                "variant_name": variant_name,
                "model_name": model_name,
                "split_name": split_name,
                "image_path": str(row.get("image_path", "")),
                "label_path": str(row.get("label_path", "")),
                "drone_type": str(row.get("drone_type", "")),
                "jitter_name": str(row.get("jitter_name", "")),
                "weather": str(row.get("weather", "")),
                "time_of_day": str(row.get("time_of_day", "")),
                "true_distance_m": true_distance,
                "predicted_distance_m": pred,
                "signed_error_m": pred - true_distance,
                "absolute_error_m": abs_error,
                "relative_error": abs_error / abs(true_distance),
            }
        )
    return output


def _metric_rows(
    variant_name: str,
    model_name: str,
    split_name: str,
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
) -> List[Dict[str, Any]]:
    overall = evaluate_predictions(rows, predictions)["overall"]
    return [
        {
            "variant_name": variant_name,
            "model_name": model_name,
            "split_name": split_name,
            **overall,
        }
    ]


def _group_metric_rows(
    prediction_rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        key = tuple(str(row[key]) for key in ["variant_name", "model_name", "split_name", *group_keys])
        grouped[key].append(row)

    output: List[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        true_values = np.asarray([float(row["true_distance_m"]) for row in group_rows], dtype=np.float64)
        predictions = np.asarray([float(row["predicted_distance_m"]) for row in group_rows], dtype=np.float64)
        metrics = evaluate_predictions(group_rows, predictions)["overall"]
        out_row: Dict[str, Any] = {
            "variant_name": key[0],
            "model_name": key[1],
            "split_name": key[2],
        }
        for index, group_key in enumerate(group_keys, start=3):
            out_row[group_key] = key[index]
        out_row["mean_true_distance_m"] = float(np.mean(true_values)) if true_values.size else 0.0
        out_row["mean_predicted_distance_m"] = float(np.mean(predictions)) if predictions.size else 0.0
        out_row.update(metrics)
        output.append(out_row)
    return output


def _feature_importance_rows(
    *,
    variant_name: str,
    feature_names: Sequence[str],
    rf_model: Any,
    xgb_model: Any,
    rf_weight: float,
    xgb_weight: float,
) -> List[Dict[str, Any]]:
    rf_values = [float(value) for value in rf_model.feature_importances_]
    xgb_values = [float(value) for value in xgb_model.feature_importances_]
    rows: List[Dict[str, Any]] = []
    for feature_name, rf_value, xgb_value in zip(feature_names, rf_values, xgb_values):
        if feature_name.endswith("__object_depth"):
            bucket = "relative_depth"
        elif "=" in feature_name:
            bucket = "metadata"
        else:
            bucket = "geometry"
        rows.append(
            {
                "variant_name": variant_name,
                "feature_name": feature_name,
                "feature_bucket": bucket,
                "rf_importance": rf_value,
                "xgb_importance": xgb_value,
                "blend_weighted_importance": (rf_weight * rf_value) + (xgb_weight * xgb_value),
            }
        )
    return sorted(rows, key=lambda row: float(row["blend_weighted_importance"]), reverse=True)


def _bucket_importance_rows(importance_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], float] = defaultdict(float)
    for row in importance_rows:
        grouped[(str(row["variant_name"]), str(row["feature_bucket"]))] += float(row["blend_weighted_importance"])
    return [
        {
            "variant_name": variant_name,
            "feature_bucket": bucket,
            "blend_weighted_importance": value,
        }
        for (variant_name, bucket), value in sorted(grouped.items())
    ]


def _plot_real_overall(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    real_rows = [row for row in rows if row["split_name"] == "real_external" and row["model_name"] == "blend"]
    labels = [str(row["variant_name"]) for row in real_rows]
    mae = [float(row["mae"]) for row in real_rows]
    mre = [100.0 * float(row["mean_relative_error"]) for row in real_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(x, mae, color="#4e79a7")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Real MAE (m)")
    axes[0].set_title("Real External MAE by Variant")
    axes[1].bar(x, mre, color="#e15759")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("Real Mean Relative Error (%)")
    axes[1].set_title("Real External Relative Error by Variant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_synthetic_vs_real(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    by_key = {(row["variant_name"], row["split_name"]): row for row in rows if row["model_name"] == "blend"}
    variants = sorted({str(row["variant_name"]) for row in rows})
    synthetic = [float(by_key[(variant, "synthetic_holdout")]["mae"]) for variant in variants]
    real = [float(by_key[(variant, "real_external")]["mae"]) for variant in variants]
    x = np.arange(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, synthetic, width=width, label="Synthetic holdout", color="#59a14f")
    ax.bar(x + width / 2, real, width=width, label="Real external", color="#e15759")
    ax.set_xticks(x, variants, rotation=30, ha="right")
    ax.set_ylabel("MAE (m)")
    ax.set_title("Synthetic vs Real MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_real_by_drone(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    filtered = [row for row in rows if row["split_name"] == "real_external" and row["model_name"] == "blend"]
    variants = sorted({str(row["variant_name"]) for row in filtered})
    drones = sorted({str(row["drone_type"]) for row in filtered})
    x = np.arange(len(variants))
    width = 0.8 / max(len(drones), 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"Kongsberg": "#b85c45", "Vestfold": "#315f8c"}
    for idx, drone_type in enumerate(drones):
        values = [
            next(float(row["mae"]) for row in filtered if row["variant_name"] == variant and row["drone_type"] == drone_type)
            for variant in variants
        ]
        offset = idx - ((len(drones) - 1) / 2.0)
        ax.bar(x + offset * width, values, width=width, label=drone_type, color=colors.get(drone_type, "#777777"))
    ax.set_xticks(x, variants, rotation=30, ha="right")
    ax.set_ylabel("Real MAE (m)")
    ax.set_title("Real External MAE by Drone Type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_bucket_importance(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    variants = sorted({str(row["variant_name"]) for row in rows})
    buckets = ["geometry", "relative_depth", "metadata"]
    colors = {"geometry": "#4e79a7", "relative_depth": "#f28e2b", "metadata": "#76b7b2"}
    bottoms = np.zeros(len(variants), dtype=np.float64)
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(14, 6))
    for bucket in buckets:
        values = [
            sum(float(row["blend_weighted_importance"]) for row in rows if row["variant_name"] == variant and row["feature_bucket"] == bucket)
            for variant in variants
        ]
        ax.bar(x, values, bottom=bottoms, label=bucket, color=colors[bucket])
        bottoms += np.asarray(values, dtype=np.float64)
    ax.set_xticks(x, variants, rotation=30, ha="right")
    ax.set_ylabel("Blend-weighted importance")
    ax.set_title("Feature Bucket Importances by Variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_winner_true_vs_predicted(
    prediction_rows: Sequence[Dict[str, Any]],
    winner_variant_name: str,
    output_path: Path,
) -> None:
    rows = [
        row for row in prediction_rows
        if row["variant_name"] == winner_variant_name and row["split_name"] == "real_external" and row["model_name"] == "blend"
    ]
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    predicted_values = np.asarray([float(row["predicted_distance_m"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 7))
    for drone_type, color in [("Kongsberg", "#b85c45"), ("Vestfold", "#315f8c")]:
        mask_rows = [row for row in rows if row["drone_type"] == drone_type]
        if not mask_rows:
            continue
        ax.scatter(
            [float(row["true_distance_m"]) for row in mask_rows],
            [float(row["predicted_distance_m"]) for row in mask_rows],
            s=24,
            alpha=0.6,
            color=color,
            label=drone_type,
        )
    line_min = float(min(true_values.min(), predicted_values.min()))
    line_max = float(max(true_values.max(), predicted_values.max()))
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black")
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Predicted Distance (m)")
    ax.set_title("Best Real Variant: True vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_study(config_path: Optional[Path] = None, dry_run: bool = False) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _read_yaml(config_path)
    sources = dict(config["sources"])
    synthetic_path = _resolve_repo_path(str(sources["synthetic_feature_table_csv"]))
    split_path = _resolve_repo_path(str(sources["synthetic_split_json"]))
    real_path = _resolve_repo_path(str(sources["real_feature_table_csv"]))
    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"

    synthetic_rows = _read_csv_rows(synthetic_path)
    real_rows = _read_csv_rows(real_path)
    split_payload = _read_json(split_path)
    train_rows = _select_rows(synthetic_rows, split_payload["dev_indices"])
    synthetic_holdout_rows = _select_rows(synthetic_rows, split_payload["test_indices"])

    raw_depth_feature_name = str(dict(config["features"])["raw_depth_feature_name"])
    all_depth_feature_names = _depth_feature_names(synthetic_rows)
    categorical_feature_names = [str(name) for name in dict(config["features"])["categorical_feature_names"]]
    categorical_levels = _categorical_levels(train_rows, categorical_feature_names)

    variants = [dict(row) for row in list(config["variants"])]
    real_columns = set(real_rows[0].keys()) if real_rows else set()
    supported_variants: List[Dict[str, Any]] = []
    skipped_variants: List[Dict[str, Any]] = []
    for variant in variants:
        feature_names = _variant_feature_names(
            variant,
            raw_depth_feature_name,
            all_depth_feature_names,
        )
        missing_real_features = [name for name in feature_names if name not in real_columns]
        if missing_real_features:
            skipped_variants.append(
                {
                    "name": str(variant["name"]),
                    "reason": "missing_real_features",
                    "missing_real_features": missing_real_features,
                }
            )
        else:
            supported_variants.append(variant)
    variants = supported_variants

    print("[attempt6-study01] Geometry-regularized synthetic-to-real study")
    print("  synthetic_train_rows: {}".format(len(train_rows)))
    print("  synthetic_holdout_rows: {}".format(len(synthetic_holdout_rows)))
    print("  real_external_rows: {}".format(len(real_rows)))
    print("  variants: {}".format(", ".join(str(row["name"]) for row in variants)))
    if skipped_variants:
        print("  skipped_variants: {}".format(", ".join(str(row["name"]) for row in skipped_variants)))
    print("  real data use: evaluation only")

    if dry_run:
        return {
            "study_name": str(config["study_name"]),
            "dry_run": True,
            "synthetic_train_rows": len(train_rows),
            "synthetic_holdout_rows": len(synthetic_holdout_rows),
            "real_external_rows": len(real_rows),
            "variant_names": [str(row["name"]) for row in variants],
            "skipped_variants": skipped_variants,
            "all_depth_feature_names": list(all_depth_feature_names),
            "categorical_levels": categorical_levels,
        }

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rf_params = dict(dict(config["model"])["rf"])
    xgb_params = dict(dict(config["model"])["xgb"])
    blend = dict(dict(config["model"])["blend"])
    rf_weight = float(blend["rf_weight"])
    xgb_weight = float(blend["xgb_weight"])

    medians = _geometry_medians(train_rows)
    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []

    for variant_index, variant in enumerate(variants, start=1):
        variant_name = str(variant["name"])
        continuous_feature_names = _variant_feature_names(
            variant,
            raw_depth_feature_name,
            all_depth_feature_names,
        )
        tabular_feature_names = _feature_names(
            continuous_feature_names,
            categorical_feature_names,
            categorical_levels,
        )
        geometry_feature_names = [name for name in continuous_feature_names if name in GEOMETRY_FEATURE_NAMES]
        corruption = dict(variant.get("train_corruption", {}))
        dropout_probability = float(corruption.get("dropout_probability", 0.0))
        noise_scale_std = float(corruption.get("noise_scale_std", 0.0))

        print("[attempt6-study01] Variant {}/{} | {}".format(variant_index, len(variants), variant_name))
        train_variant_rows = _corrupt_training_rows(
            train_rows,
            geometry_feature_names=geometry_feature_names,
            dropout_probability=dropout_probability,
            noise_scale_std=noise_scale_std,
            medians=medians,
            seed=20260520 + variant_index,
        )

        x_train = _matrix(train_variant_rows, continuous_feature_names, categorical_feature_names, categorical_levels)
        y_train = _target_values(train_variant_rows)
        x_synth = _matrix(synthetic_holdout_rows, continuous_feature_names, categorical_feature_names, categorical_levels)
        x_real = _matrix(real_rows, continuous_feature_names, categorical_feature_names, categorical_levels)

        rf_model = _fit_rf(rf_params)
        rf_model.fit(x_train, y_train)

        xgb_model = _fit_xgb(xgb_params)
        xgb_model.fit(x_train, y_train)

        for split_name, rows_for_split, x_split in [
            ("synthetic_holdout", synthetic_holdout_rows, x_synth),
            ("real_external", real_rows, x_real),
        ]:
            rf_predictions = np.asarray(rf_model.predict(x_split), dtype=np.float64)
            xgb_predictions = np.asarray(xgb_model.predict(x_split), dtype=np.float64)
            blend_predictions = (rf_weight * rf_predictions) + (xgb_weight * xgb_predictions)

            for model_name, predictions in [
                ("rf", rf_predictions),
                ("xgb", xgb_predictions),
                ("blend", blend_predictions),
            ]:
                metric_rows.extend(_metric_rows(variant_name, model_name, split_name, rows_for_split, predictions))
                prediction_rows.extend(
                    _prediction_rows(
                        rows_for_split,
                        predictions,
                        variant_name=variant_name,
                        model_name=model_name,
                        split_name=split_name,
                    )
                )

        importance_rows.extend(
            _feature_importance_rows(
                variant_name=variant_name,
                feature_names=tabular_feature_names,
                rf_model=rf_model,
                xgb_model=xgb_model,
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
            )
        )

        real_blend = next(
            row for row in metric_rows
            if row["variant_name"] == variant_name and row["model_name"] == "blend" and row["split_name"] == "real_external"
        )
        print(
            "  real blend MAE={:.4f} | MRE={:.4f} | overprediction={:.4f}".format(
                float(real_blend["mae"]),
                float(real_blend["mean_relative_error"]),
                float(real_blend["overprediction_rate"]),
            )
        )

    real_predictions = [row for row in prediction_rows if row["split_name"] == "real_external"]
    real_by_drone = _group_metric_rows(real_predictions, ["drone_type"])
    real_by_distance = _group_metric_rows(real_predictions, ["true_distance_m"])
    bucket_rows = _bucket_importance_rows(importance_rows)

    _write_csv_rows(reports_dir / "variant_metrics.csv", metric_rows)
    _write_csv_rows(reports_dir / "predictions.csv", prediction_rows)
    _write_csv_rows(reports_dir / "real_by_drone_type.csv", real_by_drone)
    _write_csv_rows(reports_dir / "real_by_distance.csv", real_by_distance)
    _write_csv_rows(reports_dir / "feature_importances.csv", importance_rows)
    _write_csv_rows(reports_dir / "feature_bucket_importances.csv", bucket_rows)

    real_blend_rows = [
        row for row in metric_rows
        if row["split_name"] == "real_external" and row["model_name"] == "blend"
    ]
    synthetic_blend_rows = [
        row for row in metric_rows
        if row["split_name"] == "synthetic_holdout" and row["model_name"] == "blend"
    ]
    winner_row = min(real_blend_rows, key=lambda row: float(row["mae"]))
    winner_variant_name = str(winner_row["variant_name"])

    _plot_real_overall(metric_rows, plots_dir / "real_external_overall_by_variant.png")
    _plot_synthetic_vs_real(metric_rows, plots_dir / "synthetic_vs_real_mae_by_variant.png")
    _plot_real_by_drone(real_by_drone, plots_dir / "real_mae_by_drone_type_and_variant.png")
    _plot_bucket_importance(bucket_rows, plots_dir / "feature_bucket_importances_by_variant.png")
    _plot_winner_true_vs_predicted(real_predictions, winner_variant_name, plots_dir / "winner_real_true_vs_predicted.png")

    summary = {
        "study_name": str(config["study_name"]),
        "config_path": str(config_path.resolve()),
        "synthetic_feature_table_csv": str(synthetic_path),
        "real_feature_table_csv": str(real_path),
        "num_synthetic_train_rows": len(train_rows),
        "num_synthetic_holdout_rows": len(synthetic_holdout_rows),
        "num_real_external_rows": len(real_rows),
        "raw_depth_feature_name": raw_depth_feature_name,
        "all_depth_feature_names": list(all_depth_feature_names),
        "categorical_levels_from_synthetic_train": categorical_levels,
        "real_data_usage": "external evaluation only; no real rows are used for fitting",
        "skipped_variants": skipped_variants,
        "rf_weight": rf_weight,
        "xgb_weight": xgb_weight,
        "winner_by_real_blend_mae": winner_row,
        "synthetic_blend_metrics": synthetic_blend_rows,
        "real_blend_metrics": real_blend_rows,
        "reports": {
            "variant_metrics_csv": str(reports_dir / "variant_metrics.csv"),
            "predictions_csv": str(reports_dir / "predictions.csv"),
            "real_by_drone_type_csv": str(reports_dir / "real_by_drone_type.csv"),
            "real_by_distance_csv": str(reports_dir / "real_by_distance.csv"),
            "feature_importances_csv": str(reports_dir / "feature_importances.csv"),
            "feature_bucket_importances_csv": str(reports_dir / "feature_bucket_importances.csv"),
        },
        "plots": {
            "real_external_overall_by_variant_png": str(plots_dir / "real_external_overall_by_variant.png"),
            "synthetic_vs_real_mae_by_variant_png": str(plots_dir / "synthetic_vs_real_mae_by_variant.png"),
            "real_mae_by_drone_type_and_variant_png": str(plots_dir / "real_mae_by_drone_type_and_variant.png"),
            "feature_bucket_importances_by_variant_png": str(plots_dir / "feature_bucket_importances_by_variant.png"),
            "winner_real_true_vs_predicted_png": str(plots_dir / "winner_real_true_vs_predicted.png"),
        },
    }
    _write_json(reports_dir / "summary.json", summary)

    analysis_lines = [
        "# Attempt6 Study01 Analysis",
        "",
        "This study trains only on synthetic rows and uses the real Nenrus rows as an external test.",
        "",
        "## Winner on Real External Blend MAE",
        "",
        "- Variant: `{}`".format(winner_variant_name),
        "- Real MAE: `{:.4f}m`".format(float(winner_row["mae"])),
        "- Real mean relative error: `{:.4f}`".format(float(winner_row["mean_relative_error"])),
        "- Real overprediction rate: `{:.4f}`".format(float(winner_row["overprediction_rate"])),
        "",
        "## Important Note",
        "",
        "The real dataset is not used to fit any RF, XGB, categorical levels, dropout medians, or noise parameters.",
    ]
    (reports_dir / "analysis.md").write_text("\n".join(analysis_lines), encoding="utf-8")

    print("[attempt6-study01] Study complete")
    print("  winner_variant: {}".format(winner_variant_name))
    print("  winner_real_mae: {:.4f}".format(float(winner_row["mae"])))
    print("  summary_json: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    result = run_study(config_path=args.config, dry_run=bool(args.dry_run))
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
