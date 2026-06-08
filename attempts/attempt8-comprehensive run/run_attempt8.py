"""
Unified synthetic rerun for the poster models.

This script rebuilds the synthetic comparison under one shared benchmark,
without touching the Nenrus calibration studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study01.run_study_01 import (  # noqa: E402
    _attach_prediction_feature,
    _cross_fit_lower_model,
    _merge_feature_sources,
    _select_lower_model_feature_names,
)
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions  # noqa: E402
from pipeline.calibration.linear_models import (  # noqa: E402
    GEOMETRY_FEATURE_NAMES,
    fit_custom_linear_model,
    fit_depth_geometry_metadata_linear_model,
    fit_depth_only_linear_model,
    fit_scale_only_model,
)
from pipeline.calibration.splits import make_distance_range_label  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
AGGREGATED_META_COLUMNS = {
    "image_path",
    "label_path",
    "stratum_key",
    "true_distance_m",
    "weather",
    "time_of_day",
}
EXACT_CATEGORY_FEATURES = ["weather", "time_of_day"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the attempt8 comprehensive synthetic rerun.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
        raise ValueError("Config did not contain a mapping: {}".format(path))
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


def _normalize_path(path_value: str) -> str:
    return str(Path(path_value).resolve())


def _select_rows(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [rows[int(index)] for index in indices]


def _load_feature_names(selection_csv: Path, source_model_name: str) -> List[str]:
    rows = _read_csv_rows(selection_csv)
    matching = [row for row in rows if str(row.get("model_name", "")) == source_model_name]
    if not matching:
        raise ValueError(
            "Could not find source model '{}' in '{}'.".format(source_model_name, selection_csv)
        )
    raw_value = str(matching[0]["feature_names"])
    feature_names = [item.strip() for item in raw_value.split("|") if item.strip()]
    if not feature_names:
        raise ValueError("Selected feature list was empty for '{}'.".format(source_model_name))
    return feature_names


def _load_aggregated_rows(feature_table_csv: Path, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    raw_rows = _read_csv_rows(feature_table_csv)
    if not raw_rows:
        raise ValueError("Aggregated feature table is empty: {}".format(feature_table_csv))

    one_hot_names = [name for name in feature_names if "=" in name]
    numeric_feature_names = [name for name in feature_names if "=" not in name]

    output_rows: List[Dict[str, Any]] = []
    for raw_row in raw_rows:
        row = dict(raw_row)
        row["source_group_id"] = _normalize_path(str(row["image_path"]))
        row["true_distance_m"] = float(row["true_distance_m"])
        for name in numeric_feature_names:
            row[name] = float(row[name])
        for name in one_hot_names:
            base_name, level = name.split("=", 1)
            row[name] = 1.0 if str(row.get(base_name, "")) == level else 0.0
        output_rows.append(row)

    missing = [name for name in feature_names if name not in output_rows[0]]
    if missing:
        raise ValueError("Aggregated rows are missing required features: {}".format(missing))
    return output_rows


def _load_expanded_rows(feature_table_csv: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(feature_table_csv)
    if not rows:
        raise ValueError("Attempt4 expanded feature table is empty: {}".format(feature_table_csv))
    output: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["source_group_id"] = _normalize_path(str(new_row["source_group_id"]))
        output.append(new_row)
    return output


def _attempt4_source_split(
    expanded_rows: Sequence[Dict[str, Any]],
    attempt4_test_split: Dict[str, Any],
    attempt4_cv_folds: Dict[str, Any],
) -> Dict[str, Any]:
    expanded_source_ids = [str(row["source_group_id"]) for row in expanded_rows]

    test_source_ids = sorted(
        {expanded_source_ids[int(index)] for index in attempt4_test_split["test_indices"]}
    )
    dev_source_ids = sorted(
        {expanded_source_ids[int(index)] for index in attempt4_test_split["dev_indices"]}
    )
    cv_folds: List[Dict[str, Any]] = []
    for fold in attempt4_cv_folds["cv_folds"]:
        cv_folds.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_source_ids": sorted(
                    {expanded_source_ids[int(index)] for index in fold["train_indices"]}
                ),
                "val_source_ids": sorted(
                    {expanded_source_ids[int(index)] for index in fold["val_indices"]}
                ),
            }
        )
    return {
        "seed": attempt4_test_split["seed"],
        "holdout_fraction": attempt4_test_split["holdout_fraction"],
        "dev_source_ids": dev_source_ids,
        "test_source_ids": test_source_ids,
        "cv_folds": cv_folds,
    }


def _build_common_benchmark(
    *,
    exact_rows: Sequence[Dict[str, Any]],
    aggregated_rows: Sequence[Dict[str, Any]],
    source_split: Dict[str, Any],
) -> Dict[str, Any]:
    exact_by_source = {str(row["source_group_id"]): dict(row) for row in exact_rows}
    aggregated_by_source = {str(row["source_group_id"]): dict(row) for row in aggregated_rows}

    split_source_ids = set(source_split["dev_source_ids"]) | set(source_split["test_source_ids"])
    exact_source_ids = set(exact_by_source.keys())
    aggregated_source_ids = set(aggregated_by_source.keys())
    common_source_ids = sorted(split_source_ids & exact_source_ids & aggregated_source_ids)

    manifest_rows: List[Dict[str, Any]] = []
    for source_id in common_source_ids:
        exact_row = exact_by_source[source_id]
        aggregated_row = aggregated_by_source[source_id]
        manifest_rows.append(
            {
                "source_group_id": source_id,
                "image_path": str(exact_row["image_path"]),
                "label_path": str(exact_row["label_path"]),
                "true_distance_m": float(exact_row["true_distance_m"]),
                "weather": str(exact_row["weather"]),
                "time_of_day": str(exact_row["time_of_day"]),
                "exact_row_present": 1,
                "aggregated_row_present": 1,
                "distance_range": make_distance_range_label(float(exact_row["true_distance_m"])),
                "aggregated_distance_match": int(
                    abs(float(aggregated_row["true_distance_m"]) - float(exact_row["true_distance_m"])) <= 1e-6
                ),
            }
        )

    source_id_to_index = {source_id: index for index, source_id in enumerate(common_source_ids)}
    exact_common_rows = [exact_by_source[source_id] for source_id in common_source_ids]
    aggregated_common_rows = [aggregated_by_source[source_id] for source_id in common_source_ids]

    dev_indices = [source_id_to_index[source_id] for source_id in source_split["dev_source_ids"] if source_id in source_id_to_index]
    test_indices = [source_id_to_index[source_id] for source_id in source_split["test_source_ids"] if source_id in source_id_to_index]
    cv_folds = []
    for fold in source_split["cv_folds"]:
        cv_folds.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_indices": [source_id_to_index[source_id] for source_id in fold["train_source_ids"] if source_id in source_id_to_index],
                "val_indices": [source_id_to_index[source_id] for source_id in fold["val_source_ids"] if source_id in source_id_to_index],
            }
        )

    dropped_rows: List[Dict[str, Any]] = []
    all_source_ids = sorted(split_source_ids | exact_source_ids | aggregated_source_ids)
    for source_id in all_source_ids:
        in_split = source_id in split_source_ids
        in_exact = source_id in exact_source_ids
        in_aggregated = source_id in aggregated_source_ids
        if in_split and in_exact and in_aggregated:
            continue
        reasons: List[str] = []
        if in_split and not in_exact:
            reasons.append("missing_exact_row")
        if in_split and not in_aggregated:
            reasons.append("missing_aggregated_row")
        if in_exact and not in_aggregated:
            reasons.append("exact_only")
        if in_aggregated and not in_exact:
            reasons.append("aggregated_only")
        if not in_split:
            reasons.append("outside_attempt4_split_universe")
        dropped_rows.append(
            {
                "source_group_id": source_id,
                "in_attempt4_split_universe": int(in_split),
                "in_exact_rows": int(in_exact),
                "in_aggregated_rows": int(in_aggregated),
                "drop_reason": "|".join(reasons),
            }
        )

    distance_counts: Dict[float, Dict[str, int]] = {}
    for row_index, row in enumerate(manifest_rows):
        distance_value = float(row["true_distance_m"])
        bucket = distance_counts.setdefault(distance_value, {"total": 0, "dev": 0, "test": 0})
        bucket["total"] += 1
        if row_index in dev_indices:
            bucket["dev"] += 1
        if row_index in test_indices:
            bucket["test"] += 1

    return {
        "exact_rows": exact_common_rows,
        "aggregated_rows": aggregated_common_rows,
        "manifest_rows": manifest_rows,
        "dropped_rows": dropped_rows,
        "split_payload": {
            "seed": source_split["seed"],
            "holdout_fraction": source_split["holdout_fraction"],
            "dev_indices": dev_indices,
            "test_indices": test_indices,
            "cv_folds": cv_folds,
        },
        "audit_summary": {
            "num_attempt4_split_source_ids": len(split_source_ids),
            "num_exact_rows": len(exact_rows),
            "num_aggregated_rows": len(aggregated_rows),
            "num_common_rows": len(common_source_ids),
            "num_dev_rows": len(dev_indices),
            "num_test_rows": len(test_indices),
            "num_dropped_rows": len(dropped_rows),
            "distance_counts": distance_counts,
        },
    }


def _build_rf_matrix(
    rows: Sequence[Dict[str, Any]],
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Optional[Dict[str, Sequence[str]]] = None,
) -> np.ndarray:
    levels_by_feature: Dict[str, List[str]] = {}
    if categorical_levels is None:
        for feature_name in categorical_feature_names:
            levels_by_feature[feature_name] = sorted({str(row[feature_name]) for row in rows})
    else:
        for feature_name in categorical_feature_names:
            levels_by_feature[feature_name] = [str(level) for level in categorical_levels[feature_name]]
    if not rows:
        feature_count = len(continuous_feature_names) + sum(
            len(levels_by_feature[feature_name]) for feature_name in categorical_feature_names
        )
        return np.zeros((0, feature_count), dtype=np.float64)
    matrix_rows: List[List[float]] = []
    for row in rows:
        values: List[float] = [float(row[name]) for name in continuous_feature_names]
        for feature_name in categorical_feature_names:
            row_value = str(row[feature_name])
            for level in levels_by_feature[feature_name]:
                values.append(1.0 if row_value == level else 0.0)
        matrix_rows.append(values)
    return np.asarray(matrix_rows, dtype=np.float64)


def _fit_exact_rf(params: Dict[str, Any]) -> RandomForestRegressor:
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


def _matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def _target_values(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)


def _prediction_rows(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    *,
    display_name: str,
    model_name: str,
    model_family: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        pred = float(prediction)
        signed_error = pred - true_distance
        output_rows.append(
            {
                "display_name": display_name,
                "model_name": model_name,
                "model_family": model_family,
                "split_name": split_name,
                "source_group_id": str(row["source_group_id"]),
                "image_path": str(row["image_path"]),
                "label_path": str(row["label_path"]),
                "weather": str(row["weather"]),
                "time_of_day": str(row["time_of_day"]),
                "distance_range": make_distance_range_label(true_distance),
                "true_distance_m": true_distance,
                "predicted_distance_m": pred,
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
            }
        )
    return output_rows


def _fold_summary_rows(
    fold_metrics: Sequence[Dict[str, Any]],
    *,
    display_name: str,
    model_name: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in fold_metrics:
        overall = dict(fold["overall"])
        rows.append(
            {
                "display_name": display_name,
                "model_name": model_name,
                "split_name": split_name,
                "fold_index": int(fold["fold_index"]),
                **overall,
            }
        )
    return rows


def _summary_from_fold_rows(fold_metrics: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metric_names = [
        "mae",
        "mean_relative_error",
        "rmse",
        "r2",
        "median_absolute_error",
        "p90_absolute_error",
        "p95_absolute_error",
        "within_5m_rate",
        "within_10m_rate",
        "within_20m_rate",
    ]
    summary: Dict[str, float] = {}
    for metric_name in metric_names:
        values = [float(fold["overall"][metric_name]) for fold in fold_metrics]
        summary["mean_{}".format(metric_name)] = float(np.mean(values))
        summary["std_{}".format(metric_name)] = float(np.std(values))
    return summary


def _build_true_distance_summary(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    *,
    display_name: str,
    model_name: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    grouped_indices: Dict[float, List[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(float(row["true_distance_m"]), []).append(index)

    predictions_array = np.asarray(predictions, dtype=np.float64)
    output_rows: List[Dict[str, Any]] = []
    for true_distance in sorted(grouped_indices):
        indices = grouped_indices[true_distance]
        group_rows = [rows[index] for index in indices]
        group_predictions = predictions_array[indices]
        metrics = evaluate_predictions(group_rows, group_predictions)["overall"]
        output_rows.append(
            {
                "display_name": display_name,
                "model_name": model_name,
                "split_name": split_name,
                "true_distance_m": true_distance,
                **metrics,
            }
        )
    return output_rows


def _metrics_row(
    *,
    display_name: str,
    model_name: str,
    model_family: str,
    split_name: str,
    metrics: Dict[str, Any],
    role: str,
    config_name: str,
) -> Dict[str, Any]:
    return {
        "display_name": display_name,
        "model_name": model_name,
        "model_family": model_family,
        "split_name": split_name,
        "role": role,
        "config_name": config_name,
        **dict(metrics),
    }


def _run_simple_linear_model(
    *,
    display_name: str,
    model_name: str,
    exact_rows: Sequence[Dict[str, Any]],
    split_payload: Dict[str, Any],
    fit_fn,
    verbose_prefix: str,
) -> Dict[str, Any]:
    dev_rows = _select_rows(exact_rows, split_payload["dev_indices"])
    test_rows = _select_rows(exact_rows, split_payload["test_indices"])

    fold_metrics: List[Dict[str, Any]] = []
    dev_prediction_lookup: Dict[int, float] = {}
    for outer_fold_index, fold in enumerate(split_payload["cv_folds"], start=1):
        train_rows = _select_rows(exact_rows, fold["train_indices"])
        val_rows = _select_rows(exact_rows, fold["val_indices"])
        print(
            "[attempt8] {} | CV fold {}/{} | train_rows={} | val_rows={}".format(
                verbose_prefix,
                outer_fold_index,
                len(split_payload["cv_folds"]),
                len(train_rows),
                len(val_rows),
            )
        )
        model = fit_fn(train_rows)
        val_predictions = np.asarray(model.predict(val_rows), dtype=np.float64)
        fold_eval = evaluate_predictions(val_rows, val_predictions)
        fold_metrics.append(
            {
                "fold_index": int(fold["fold_index"]),
                "overall": fold_eval["overall"],
                "per_distance_range": fold_eval["per_distance_range"],
            }
        )
        for global_index, prediction in zip(fold["val_indices"], val_predictions):
            dev_prediction_lookup[int(global_index)] = float(prediction)

    dev_predictions = np.asarray(
        [dev_prediction_lookup[int(index)] for index in split_payload["dev_indices"]],
        dtype=np.float64,
    )
    cv_eval = evaluate_predictions(dev_rows, dev_predictions)

    print("[attempt8] {} | held-out test fit".format(verbose_prefix))
    model = fit_fn(dev_rows)
    test_predictions = np.asarray(model.predict(test_rows), dtype=np.float64)
    test_eval = evaluate_predictions(test_rows, test_predictions)

    return {
        "display_name": display_name,
        "model_name": model_name,
        "model_family": "linear",
        "role": "poster",
        "config_name": model_name,
        "fold_metrics": fold_metrics,
        "cv_eval": cv_eval,
        "test_eval": test_eval,
        "cv_predictions": dev_predictions,
        "test_predictions": test_predictions,
        "cv_rows": dev_rows,
        "test_rows": test_rows,
        "coefficient_rows": model.coefficient_rows(),
    }


def _run_improved_linear(
    *,
    display_name: str,
    exact_rows: Sequence[Dict[str, Any]],
    split_payload: Dict[str, Any],
    lower_feature_names: Sequence[str],
    lower_config: Dict[str, Any],
) -> Dict[str, Any]:
    dev_rows = _select_rows(exact_rows, split_payload["dev_indices"])
    test_rows = _select_rows(exact_rows, split_payload["test_indices"])
    lower_prediction_feature_name = str(lower_config["score_feature_name"])

    lower_rf_params = dict(lower_config["random_forest"])
    lower_inner_num_folds = int(lower_config["inner_num_folds"])
    lower_inner_seed = int(lower_config["inner_seed"])

    fold_metrics: List[Dict[str, Any]] = []
    dev_prediction_lookup: Dict[int, float] = {}
    lower_feature_lookup: Dict[int, float] = {}
    for outer_fold_index, fold in enumerate(split_payload["cv_folds"], start=1):
        train_rows = _select_rows(exact_rows, fold["train_indices"])
        val_rows = _select_rows(exact_rows, fold["val_indices"])
        print(
            "[attempt8] improved linear | lower cross-fit outer fold {}/{} | train_rows={} | val_rows={}".format(
                outer_fold_index,
                len(split_payload["cv_folds"]),
                len(train_rows),
                len(val_rows),
            )
        )
        lower_cache = _cross_fit_lower_model(
            train_rows=train_rows,
            lower_feature_names=lower_feature_names,
            rf_params=lower_rf_params,
            num_folds=lower_inner_num_folds,
            seed=lower_inner_seed + int(fold["fold_index"]),
            verbose=True,
            progress_label="attempt8 improved_linear outer_fold={}".format(outer_fold_index),
        )
        train_augmented = _attach_prediction_feature(
            train_rows,
            lower_prediction_feature_name,
            lower_cache["oof_predictions"],
        )
        val_lower_predictions = np.asarray(
            lower_cache["full_estimator"].predict(_matrix(val_rows, lower_feature_names)),
            dtype=np.float64,
        )
        val_augmented = _attach_prediction_feature(
            val_rows,
            lower_prediction_feature_name,
            val_lower_predictions,
        )
        model = fit_custom_linear_model(
            records=train_augmented,
            model_name="stacked_rf_depth_geometry_metadata_linear",
            continuous_feature_names=[lower_prediction_feature_name] + list(GEOMETRY_FEATURE_NAMES),
            categorical_feature_names=EXACT_CATEGORY_FEATURES,
        )
        val_predictions = np.asarray(model.predict(val_augmented), dtype=np.float64)
        fold_eval = evaluate_predictions(val_augmented, val_predictions)
        fold_metrics.append(
            {
                "fold_index": int(fold["fold_index"]),
                "overall": fold_eval["overall"],
                "per_distance_range": fold_eval["per_distance_range"],
            }
        )
        for global_index, prediction in zip(fold["val_indices"], val_predictions):
            dev_prediction_lookup[int(global_index)] = float(prediction)
        for global_index, lower_prediction in zip(fold["val_indices"], val_lower_predictions):
            lower_feature_lookup[int(global_index)] = float(lower_prediction)

    dev_predictions = np.asarray(
        [dev_prediction_lookup[int(index)] for index in split_payload["dev_indices"]],
        dtype=np.float64,
    )
    dev_lower_predictions = np.asarray(
        [lower_feature_lookup[int(index)] for index in split_payload["dev_indices"]],
        dtype=np.float64,
    )
    dev_augmented_rows = _attach_prediction_feature(dev_rows, lower_prediction_feature_name, dev_lower_predictions)
    cv_eval = evaluate_predictions(dev_augmented_rows, dev_predictions)

    print("[attempt8] improved linear | building shared held-out lower feature")
    shared_lower_cache = _cross_fit_lower_model(
        train_rows=dev_rows,
        lower_feature_names=lower_feature_names,
        rf_params=lower_rf_params,
        num_folds=lower_inner_num_folds,
        seed=lower_inner_seed + 1000,
        verbose=True,
        progress_label="attempt8 improved_linear heldout",
    )
    shared_dev_augmented_rows = _attach_prediction_feature(
        dev_rows,
        lower_prediction_feature_name,
        shared_lower_cache["oof_predictions"],
    )
    test_lower_predictions = np.asarray(
        shared_lower_cache["full_estimator"].predict(_matrix(test_rows, lower_feature_names)),
        dtype=np.float64,
    )
    shared_test_augmented_rows = _attach_prediction_feature(
        test_rows,
        lower_prediction_feature_name,
        test_lower_predictions,
    )

    print("[attempt8] improved linear | held-out test fit")
    model = fit_custom_linear_model(
        records=shared_dev_augmented_rows,
        model_name="stacked_rf_depth_geometry_metadata_linear",
        continuous_feature_names=[lower_prediction_feature_name] + list(GEOMETRY_FEATURE_NAMES),
        categorical_feature_names=EXACT_CATEGORY_FEATURES,
    )
    test_predictions = np.asarray(model.predict(shared_test_augmented_rows), dtype=np.float64)
    test_eval = evaluate_predictions(shared_test_augmented_rows, test_predictions)

    return {
        "display_name": display_name,
        "model_name": "stacked_rf_depth_geometry_metadata_linear",
        "model_family": "stacked_linear",
        "role": "poster",
        "config_name": str(lower_config["selected_model_name"]),
        "fold_metrics": fold_metrics,
        "cv_eval": cv_eval,
        "test_eval": test_eval,
        "cv_predictions": dev_predictions,
        "test_predictions": test_predictions,
        "cv_rows": dev_augmented_rows,
        "test_rows": shared_test_augmented_rows,
        "coefficient_rows": model.coefficient_rows(),
        "lower_feature_names": list(lower_feature_names),
        "lower_feature_importance_rows": list(shared_lower_cache["feature_importance_rows"]),
    }


def _run_exact_rf(
    *,
    display_name: str,
    exact_rows: Sequence[Dict[str, Any]],
    split_payload: Dict[str, Any],
    raw_depth_feature_name: str,
    candidate_configs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    dev_rows = _select_rows(exact_rows, split_payload["dev_indices"])
    test_rows = _select_rows(exact_rows, split_payload["test_indices"])
    continuous_feature_names = [raw_depth_feature_name] + list(GEOMETRY_FEATURE_NAMES)
    categorical_levels = {
        feature_name: sorted({str(row[feature_name]) for row in exact_rows})
        for feature_name in EXACT_CATEGORY_FEATURES
    }
    dev_index_lookup = {
        int(global_index): local_index for local_index, global_index in enumerate(split_payload["dev_indices"])
    }
    candidate_results: List[Dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidate_configs, start=1):
        candidate_params = dict(candidate)
        candidate_name = str(candidate_params.pop("name"))
        print(
            "[attempt8] RF inner50 | candidate {}/{} | {}".format(
                candidate_index,
                len(candidate_configs),
                candidate_name,
            )
        )
        oof_predictions = np.zeros(len(dev_rows), dtype=np.float64)
        fold_metrics: List[Dict[str, Any]] = []
        for outer_fold_index, fold in enumerate(split_payload["cv_folds"], start=1):
            train_rows = _select_rows(exact_rows, fold["train_indices"])
            val_rows = _select_rows(exact_rows, fold["val_indices"])
            estimator = _fit_exact_rf(candidate_params)
            estimator.fit(
                _build_rf_matrix(
                    train_rows,
                    continuous_feature_names,
                    EXACT_CATEGORY_FEATURES,
                    categorical_levels,
                ),
                _target_values(train_rows),
            )
            val_predictions = np.asarray(
                estimator.predict(
                    _build_rf_matrix(
                        val_rows,
                        continuous_feature_names,
                        EXACT_CATEGORY_FEATURES,
                        categorical_levels,
                    )
                ),
                dtype=np.float64,
            )
            local_val_indices = [dev_index_lookup[int(index)] for index in fold["val_indices"]]
            for local_index, prediction in zip(local_val_indices, val_predictions):
                oof_predictions[int(local_index)] = float(prediction)
            fold_eval = evaluate_predictions(val_rows, val_predictions)
            fold_metrics.append(
                {
                    "fold_index": int(fold["fold_index"]),
                    "overall": fold_eval["overall"],
                    "per_distance_range": fold_eval["per_distance_range"],
                }
            )
            print(
                "[attempt8] RF inner50 | {} | fold {}/{} | fold_mae={:.4f}".format(
                    candidate_name,
                    outer_fold_index,
                    len(split_payload["cv_folds"]),
                    float(fold_eval["overall"]["mae"]),
                )
            )
        oof_eval = evaluate_predictions(dev_rows, oof_predictions)
        candidate_results.append(
            {
                "candidate_name": candidate_name,
                "candidate_params": candidate_params,
                "fold_metrics": fold_metrics,
                "oof_predictions": oof_predictions,
                "oof_eval": oof_eval,
            }
        )
        print(
            "[attempt8] RF inner50 | {} | oof_mae={:.4f}".format(
                candidate_name,
                float(oof_eval["overall"]["mae"]),
            )
        )

    best_candidate = sorted(
        candidate_results,
        key=lambda row: (
            float(row["oof_eval"]["overall"]["mae"]),
            float(row["oof_eval"]["overall"]["mean_relative_error"]),
            str(row["candidate_name"]),
        ),
    )[0]
    print("[attempt8] RF inner50 | selected candidate={}".format(best_candidate["candidate_name"]))

    estimator = _fit_exact_rf(best_candidate["candidate_params"])
    estimator.fit(
        _build_rf_matrix(dev_rows, continuous_feature_names, EXACT_CATEGORY_FEATURES, categorical_levels),
        _target_values(dev_rows),
    )
    test_predictions = np.asarray(
        estimator.predict(
            _build_rf_matrix(test_rows, continuous_feature_names, EXACT_CATEGORY_FEATURES, categorical_levels)
        ),
        dtype=np.float64,
    )
    test_eval = evaluate_predictions(test_rows, test_predictions)
    importance_rows = []
    feature_names = list(continuous_feature_names)
    for feature_name in EXACT_CATEGORY_FEATURES:
        for level in categorical_levels[feature_name]:
            feature_names.append("{}={}".format(feature_name, level))
    for feature_name, importance_value in zip(feature_names, estimator.feature_importances_):
        importance_rows.append(
            {
                "display_name": display_name,
                "model_name": "raw_depth_geometry_metadata_rf",
                "candidate_name": best_candidate["candidate_name"],
                "term": feature_name,
                "importance_value": float(importance_value),
            }
        )

    return {
        "display_name": display_name,
        "model_name": "raw_depth_geometry_metadata_rf",
        "model_family": "random_forest",
        "role": "poster",
        "config_name": best_candidate["candidate_name"],
        "fold_metrics": best_candidate["fold_metrics"],
        "cv_eval": best_candidate["oof_eval"],
        "test_eval": test_eval,
        "cv_predictions": best_candidate["oof_predictions"],
        "test_predictions": test_predictions,
        "cv_rows": dev_rows,
        "test_rows": test_rows,
        "candidate_rows": [
            {
                "display_name": display_name,
                "model_name": "raw_depth_geometry_metadata_rf",
                "candidate_name": row["candidate_name"],
                "oof_mae": float(row["oof_eval"]["overall"]["mae"]),
                "oof_mean_relative_error": float(row["oof_eval"]["overall"]["mean_relative_error"]),
                "oof_rmse": float(row["oof_eval"]["overall"]["rmse"]),
                "oof_r2": float(row["oof_eval"]["overall"]["r2"]),
                **{"param_{}".format(key): value for key, value in row["candidate_params"].items()},
            }
            for row in candidate_results
        ],
        "feature_importance_rows": importance_rows,
    }


def _evaluate_aggregated_candidate_oof(
    *,
    model_family: str,
    candidate_name: str,
    params: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    cv_folds: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    oof_predictions = np.zeros(len(rows), dtype=np.float64)
    for fold in cv_folds:
        train_rows = _select_rows(rows, fold["train_indices"])
        val_rows = _select_rows(rows, fold["val_indices"])
        estimator = _fit_exact_rf(params) if model_family == "rf" else _fit_xgb(params)
        estimator.fit(_matrix(train_rows, feature_names), _target_values(train_rows))
        fold_predictions = np.asarray(estimator.predict(_matrix(val_rows, feature_names)), dtype=np.float64)
        local_val_indices = [split_payload_index for split_payload_index in fold["val_indices"]]
        for local_index, prediction in zip(local_val_indices, fold_predictions):
            oof_predictions[int(local_index)] = float(prediction)
    return {
        "candidate_name": candidate_name,
        "params": dict(params),
        "oof_predictions": oof_predictions,
        "oof_eval": evaluate_predictions(rows, oof_predictions),
    }


def _fit_selected_aggregated_model(
    *,
    model_family: str,
    params: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    estimator = _fit_exact_rf(params) if model_family == "rf" else _fit_xgb(params)
    estimator.fit(_matrix(train_rows, feature_names), _target_values(train_rows))
    return np.asarray(estimator.predict(_matrix(test_rows, feature_names)), dtype=np.float64)


def _blend_predictions(
    rf_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
    *,
    rf_weight: float,
    clip_min_distance_m: float,
) -> np.ndarray:
    blended = (rf_weight * rf_predictions) + ((1.0 - rf_weight) * xgb_predictions)
    return np.maximum(blended, float(clip_min_distance_m))


def _run_aggregated_family(
    *,
    aggregated_rows: Sequence[Dict[str, Any]],
    split_payload: Dict[str, Any],
    feature_names: Sequence[str],
    rf_candidates: Sequence[Dict[str, Any]],
    xgb_candidates: Sequence[Dict[str, Any]],
    blend_config: Dict[str, Any],
) -> Dict[str, Any]:
    dev_rows = _select_rows(aggregated_rows, split_payload["dev_indices"])
    test_rows = _select_rows(aggregated_rows, split_payload["test_indices"])

    cv_folds = []
    dev_index_lookup = {int(global_index): local_index for local_index, global_index in enumerate(split_payload["dev_indices"])}
    for fold in split_payload["cv_folds"]:
        cv_folds.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_indices": [dev_index_lookup[int(index)] for index in fold["train_indices"]],
                "val_indices": [dev_index_lookup[int(index)] for index in fold["val_indices"]],
            }
        )

    rf_results: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(rf_candidates, start=1):
        candidate_params = dict(candidate)
        candidate_name = str(candidate_params.pop("name"))
        print("[attempt8] RF+jitter | RF candidate {}/{} | {}".format(candidate_index, len(rf_candidates), candidate_name))
        result = _evaluate_aggregated_candidate_oof(
            model_family="rf",
            candidate_name=candidate_name,
            params=candidate_params,
            rows=dev_rows,
            feature_names=feature_names,
            cv_folds=cv_folds,
        )
        rf_results.append(result)
        print("[attempt8] RF+jitter | {} | oof_mae={:.4f}".format(candidate_name, float(result["oof_eval"]["overall"]["mae"])))

    xgb_results: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(xgb_candidates, start=1):
        candidate_params = dict(candidate)
        candidate_name = str(candidate_params.pop("name"))
        print("[attempt8] ensemble | XGB candidate {}/{} | {}".format(candidate_index, len(xgb_candidates), candidate_name))
        result = _evaluate_aggregated_candidate_oof(
            model_family="xgb",
            candidate_name=candidate_name,
            params=candidate_params,
            rows=dev_rows,
            feature_names=feature_names,
            cv_folds=cv_folds,
        )
        xgb_results.append(result)
        print("[attempt8] ensemble | {} | oof_mae={:.4f}".format(candidate_name, float(result["oof_eval"]["overall"]["mae"])))

    best_rf = sorted(
        rf_results,
        key=lambda row: (
            float(row["oof_eval"]["overall"]["mae"]),
            float(row["oof_eval"]["overall"]["mean_relative_error"]),
            str(row["candidate_name"]),
        ),
    )[0]
    best_xgb = sorted(
        xgb_results,
        key=lambda row: (
            float(row["oof_eval"]["overall"]["mae"]),
            float(row["oof_eval"]["overall"]["mean_relative_error"]),
            str(row["candidate_name"]),
        ),
    )[0]
    print("[attempt8] aggregated family | selected RF={} | XGB={}".format(best_rf["candidate_name"], best_xgb["candidate_name"]))

    rf_test_predictions = _fit_selected_aggregated_model(
        model_family="rf",
        params=best_rf["params"],
        train_rows=dev_rows,
        test_rows=test_rows,
        feature_names=feature_names,
    )
    xgb_test_predictions = _fit_selected_aggregated_model(
        model_family="xgb",
        params=best_xgb["params"],
        train_rows=dev_rows,
        test_rows=test_rows,
        feature_names=feature_names,
    )

    weight_grid_step = float(blend_config["weight_grid_step"])
    clip_min_distance_m = float(blend_config.get("clip_min_distance_m", 0.0))
    candidate_weights = sorted({round(index * weight_grid_step, 10) for index in range(int(round(1.0 / weight_grid_step)) + 1)} | {1.0})
    blend_search_rows: List[Dict[str, Any]] = []
    best_blend_row: Optional[Dict[str, Any]] = None
    for rf_weight in candidate_weights:
        predictions = _blend_predictions(
            np.asarray(best_rf["oof_predictions"], dtype=np.float64),
            np.asarray(best_xgb["oof_predictions"], dtype=np.float64),
            rf_weight=float(rf_weight),
            clip_min_distance_m=clip_min_distance_m,
        )
        overall = evaluate_predictions(dev_rows, predictions)["overall"]
        row = {
            "rf_weight": float(rf_weight),
            "xgb_weight": float(1.0 - rf_weight),
            "oof_mae": float(overall["mae"]),
            "oof_mean_relative_error": float(overall["mean_relative_error"]),
            "oof_rmse": float(overall["rmse"]),
            "oof_r2": float(overall["r2"]),
        }
        blend_search_rows.append(row)
        if best_blend_row is None or (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            float(row["rf_weight"]),
        ) < (
            float(best_blend_row["oof_mae"]),
            float(best_blend_row["oof_mean_relative_error"]),
            float(best_blend_row["rf_weight"]),
        ):
            best_blend_row = row

    assert best_blend_row is not None
    print(
        "[attempt8] ensemble | selected blend rf_weight={:.2f} | xgb_weight={:.2f} | oof_mae={:.4f}".format(
            float(best_blend_row["rf_weight"]),
            float(best_blend_row["xgb_weight"]),
            float(best_blend_row["oof_mae"]),
        )
    )

    family_specs = [
        {
            "display_name": "RF+jitter",
            "model_name": "rf_winner_baseline",
            "model_family": "aggregated_rf",
            "role": "poster",
            "config_name": str(best_rf["candidate_name"]),
            "rf_weight": 1.0,
            "xgb_weight": 0.0,
        },
        {
            "display_name": "XGBoost+jitter",
            "model_name": "xgb_winner_baseline",
            "model_family": "aggregated_xgb",
            "role": "internal",
            "config_name": str(best_xgb["candidate_name"]),
            "rf_weight": 0.0,
            "xgb_weight": 1.0,
        },
        {
            "display_name": "ensemble",
            "model_name": "tuned_weight_blend",
            "model_family": "aggregated_blend",
            "role": "poster",
            "config_name": "rf_{:.2f}_xgb_{:.2f}".format(
                float(best_blend_row["rf_weight"]),
                float(best_blend_row["xgb_weight"]),
            ),
            "rf_weight": float(best_blend_row["rf_weight"]),
            "xgb_weight": float(best_blend_row["xgb_weight"]),
        },
    ]

    results: List[Dict[str, Any]] = []
    for spec in family_specs:
        cv_predictions = _blend_predictions(
            np.asarray(best_rf["oof_predictions"], dtype=np.float64),
            np.asarray(best_xgb["oof_predictions"], dtype=np.float64),
            rf_weight=float(spec["rf_weight"]),
            clip_min_distance_m=clip_min_distance_m,
        )
        test_predictions = _blend_predictions(
            np.asarray(rf_test_predictions, dtype=np.float64),
            np.asarray(xgb_test_predictions, dtype=np.float64),
            rf_weight=float(spec["rf_weight"]),
            clip_min_distance_m=clip_min_distance_m,
        )
        cv_eval = evaluate_predictions(dev_rows, cv_predictions)
        test_eval = evaluate_predictions(test_rows, test_predictions)
        print(
            "[attempt8] {} | cv_mae={:.4f} | test_mae={:.4f}".format(
                spec["display_name"],
                float(cv_eval["overall"]["mae"]),
                float(test_eval["overall"]["mae"]),
            )
        )
        results.append(
            {
                "display_name": spec["display_name"],
                "model_name": spec["model_name"],
                "model_family": spec["model_family"],
                "role": spec["role"],
                "config_name": spec["config_name"],
                "rf_weight": float(spec["rf_weight"]),
                "xgb_weight": float(spec["xgb_weight"]),
                "cv_eval": cv_eval,
                "test_eval": test_eval,
                "cv_predictions": cv_predictions,
                "test_predictions": test_predictions,
                "cv_rows": dev_rows,
                "test_rows": test_rows,
            }
        )

    return {
        "results": results,
        "rf_candidate_rows": [
            {
                "display_name": "RF+jitter",
                "model_name": "rf_winner_baseline",
                "candidate_name": row["candidate_name"],
                "oof_mae": float(row["oof_eval"]["overall"]["mae"]),
                "oof_mean_relative_error": float(row["oof_eval"]["overall"]["mean_relative_error"]),
                "oof_rmse": float(row["oof_eval"]["overall"]["rmse"]),
                "oof_r2": float(row["oof_eval"]["overall"]["r2"]),
                **{"param_{}".format(key): value for key, value in row["params"].items()},
            }
            for row in rf_results
        ],
        "xgb_candidate_rows": [
            {
                "display_name": "XGBoost+jitter",
                "model_name": "xgb_winner_baseline",
                "candidate_name": row["candidate_name"],
                "oof_mae": float(row["oof_eval"]["overall"]["mae"]),
                "oof_mean_relative_error": float(row["oof_eval"]["overall"]["mean_relative_error"]),
                "oof_rmse": float(row["oof_eval"]["overall"]["rmse"]),
                "oof_r2": float(row["oof_eval"]["overall"]["r2"]),
                **{"param_{}".format(key): value for key, value in row["params"].items()},
            }
            for row in xgb_results
        ],
        "blend_search_rows": blend_search_rows,
        "best_rf": best_rf,
        "best_xgb": best_xgb,
        "best_blend_row": best_blend_row,
    }


def run_study(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _read_yaml(config_path)
    output_root = _resolve_repo_path(str(config["output_root"]))
    audit_dir = output_root / "audit"
    features_dir = output_root / "features"
    splits_dir = output_root / "splits"
    reports_dir = output_root / "reports"
    for path in (audit_dir, features_dir, splits_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("[attempt8] Loading cached source tables")
    sources = dict(config["sources"])
    attempt1_features_csv = _resolve_repo_path(str(sources["attempt1_features_csv"]))
    study06_fused_features_csv = _resolve_repo_path(str(sources["study06_fused_features_csv"]))
    study06_metrics_csv = _resolve_repo_path(str(sources["study06_metrics_csv"]))
    aggregated_feature_table_csv = _resolve_repo_path(str(sources["aggregated_feature_table_csv"]))
    aggregated_feature_selection_csv = _resolve_repo_path(str(sources["aggregated_feature_selection_csv"]))
    attempt4_expanded_feature_table_csv = _resolve_repo_path(str(sources["attempt4_expanded_feature_table_csv"]))
    attempt4_test_split_json = _resolve_repo_path(str(sources["attempt4_test_split_json"]))
    attempt4_cv_folds_json = _resolve_repo_path(str(sources["attempt4_cv_folds_json"]))

    exact_rows, study06_feature_names, exact_merge_summary = _merge_feature_sources(
        attempt1_features_csv=attempt1_features_csv,
        study06_fused_features_csv=study06_fused_features_csv,
    )
    for row in exact_rows:
        row["source_group_id"] = _normalize_path(str(row["image_path"]))

    lower_config = dict(config["lower_model"])
    selected_lower_model_name, selected_lower_feature_names, selected_lower_model_row = _select_lower_model_feature_names(
        metrics_csv_path=study06_metrics_csv,
        requested_model_name=lower_config.get("selected_model_name"),
        available_feature_names=study06_feature_names,
    )
    lower_config["selected_model_name"] = selected_lower_model_name

    aggregated_feature_names = _load_feature_names(
        aggregated_feature_selection_csv,
        str(dict(config["benchmark"])["aggregated_source_model_name"]),
    )
    aggregated_rows = _load_aggregated_rows(aggregated_feature_table_csv, aggregated_feature_names)
    expanded_rows = _load_expanded_rows(attempt4_expanded_feature_table_csv)
    source_split = _attempt4_source_split(
        expanded_rows=expanded_rows,
        attempt4_test_split=_read_json(attempt4_test_split_json),
        attempt4_cv_folds=_read_json(attempt4_cv_folds_json),
    )

    print("[attempt8] Building common benchmark intersection")
    benchmark = _build_common_benchmark(
        exact_rows=exact_rows,
        aggregated_rows=aggregated_rows,
        source_split=source_split,
    )
    exact_common_rows = benchmark["exact_rows"]
    aggregated_common_rows = benchmark["aggregated_rows"]
    split_payload = benchmark["split_payload"]

    print(
        "[attempt8] Common benchmark ready | rows={} | dev={} | test={}".format(
            len(exact_common_rows),
            len(split_payload["dev_indices"]),
            len(split_payload["test_indices"]),
        )
    )

    _write_csv_rows(features_dir / "exact_common_feature_table.csv", exact_common_rows)
    _write_csv_rows(features_dir / "aggregated_common_feature_table.csv", aggregated_common_rows)
    _write_csv_rows(audit_dir / "common_manifest.csv", benchmark["manifest_rows"])
    _write_csv_rows(audit_dir / "dropped_rows_audit.csv", benchmark["dropped_rows"])
    _write_json(audit_dir / "coverage_summary.json", {**exact_merge_summary, **benchmark["audit_summary"]})
    _write_json(splits_dir / "unified_split.json", {
        "seed": split_payload["seed"],
        "holdout_fraction": split_payload["holdout_fraction"],
        "dev_indices": split_payload["dev_indices"],
        "test_indices": split_payload["test_indices"],
    })
    _write_json(splits_dir / "unified_cv_folds.json", {"cv_folds": split_payload["cv_folds"]})

    model_catalog = [
        {
            "display_name": "scale only",
            "model_name": "scale_only",
            "model_family": "linear",
            "role": "poster",
            "feature_basis": "attempt1 relative_depth only via global scale factor",
        },
        {
            "display_name": "depth linear",
            "model_name": "depth_only_linear",
            "model_family": "linear",
            "role": "poster",
            "feature_basis": "attempt1 relative_depth only",
        },
        {
            "display_name": "linear+bb",
            "model_name": "depth_geometry_metadata_linear",
            "model_family": "linear",
            "role": "poster",
            "feature_basis": "attempt1 relative_depth + bbox geometry + metadata",
        },
        {
            "display_name": "improved linear",
            "model_name": "stacked_rf_depth_geometry_metadata_linear",
            "model_family": "stacked_linear",
            "role": "poster",
            "feature_basis": "attempt2 study06 lower RF prediction + bbox geometry + metadata",
        },
        {
            "display_name": "RF",
            "model_name": "raw_depth_geometry_metadata_rf",
            "model_family": "random_forest",
            "role": "poster",
            "feature_basis": "{} + bbox geometry + metadata".format(str(dict(config["exact_rf"])["raw_depth_feature_name"])),
        },
        {
            "display_name": "RF+jitter",
            "model_name": "rf_winner_baseline",
            "model_family": "aggregated_rf",
            "role": "poster",
            "feature_basis": "aggregated jitter-summary row with inner50 jitter median + geometry medians/stds + metadata",
        },
        {
            "display_name": "XGBoost+jitter",
            "model_name": "xgb_winner_baseline",
            "model_family": "aggregated_xgb",
            "role": "internal",
            "feature_basis": "same aggregated jitter-summary row as RF+jitter",
        },
        {
            "display_name": "ensemble",
            "model_name": "tuned_weight_blend",
            "model_family": "aggregated_blend",
            "role": "poster",
            "feature_basis": "weighted blend of RF+jitter and XGBoost+jitter",
        },
    ]
    _write_csv_rows(reports_dir / "model_catalog.csv", model_catalog)

    print("[attempt8] Running exact linear baselines")
    results: List[Dict[str, Any]] = []
    results.append(
        _run_simple_linear_model(
            display_name="scale only",
            model_name="scale_only",
            exact_rows=exact_common_rows,
            split_payload=split_payload,
            fit_fn=fit_scale_only_model,
            verbose_prefix="scale only",
        )
    )
    results.append(
        _run_simple_linear_model(
            display_name="depth linear",
            model_name="depth_only_linear",
            exact_rows=exact_common_rows,
            split_payload=split_payload,
            fit_fn=fit_depth_only_linear_model,
            verbose_prefix="depth linear",
        )
    )
    results.append(
        _run_simple_linear_model(
            display_name="linear+bb",
            model_name="depth_geometry_metadata_linear",
            exact_rows=exact_common_rows,
            split_payload=split_payload,
            fit_fn=fit_depth_geometry_metadata_linear_model,
            verbose_prefix="linear+bb",
        )
    )
    results.append(
        _run_improved_linear(
            display_name="improved linear",
            exact_rows=exact_common_rows,
            split_payload=split_payload,
            lower_feature_names=selected_lower_feature_names,
            lower_config=lower_config,
        )
    )
    results.append(
        _run_exact_rf(
            display_name="RF",
            exact_rows=exact_common_rows,
            split_payload=split_payload,
            raw_depth_feature_name=str(dict(config["exact_rf"])["raw_depth_feature_name"]),
            candidate_configs=list(dict(config["exact_rf"])["candidate_configs"]),
        )
    )

    print("[attempt8] Running aggregated jitter family")
    aggregated_family = _run_aggregated_family(
        aggregated_rows=aggregated_common_rows,
        split_payload=split_payload,
        feature_names=aggregated_feature_names,
        rf_candidates=list(dict(config["aggregated_models"])["rf_candidates"]),
        xgb_candidates=list(dict(config["aggregated_models"])["xgb_candidates"]),
        blend_config=dict(config["blend"]),
    )
    results.extend(aggregated_family["results"])

    metrics_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []
    true_distance_summary_rows: List[Dict[str, Any]] = []
    distance_range_metric_rows: List[Dict[str, Any]] = []
    fold_metric_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []
    lower_model_importance_rows: List[Dict[str, Any]] = []
    exact_rf_candidate_rows: List[Dict[str, Any]] = []
    aggregated_rf_candidate_rows: List[Dict[str, Any]] = []
    xgb_candidate_rows: List[Dict[str, Any]] = []

    for result in results:
        metrics_rows.append(
            _metrics_row(
                display_name=result["display_name"],
                model_name=result["model_name"],
                model_family=result["model_family"],
                split_name="cv_oof",
                metrics=result["cv_eval"]["overall"],
                role=result["role"],
                config_name=result["config_name"],
            )
        )
        metrics_rows.append(
            _metrics_row(
                display_name=result["display_name"],
                model_name=result["model_name"],
                model_family=result["model_family"],
                split_name="test",
                metrics=result["test_eval"]["overall"],
                role=result["role"],
                config_name=result["config_name"],
            )
        )
        for split_name, eval_payload in (("cv_oof", result["cv_eval"]), ("test", result["test_eval"])):
            for distance_range, metrics in dict(eval_payload["per_distance_range"]).items():
                distance_range_metric_rows.append(
                    {
                        "display_name": result["display_name"],
                        "model_name": result["model_name"],
                        "model_family": result["model_family"],
                        "split_name": split_name,
                        "role": result["role"],
                        "config_name": result["config_name"],
                        "distance_range": distance_range,
                        **dict(metrics),
                    }
                )
        prediction_rows.extend(
            _prediction_rows(
                result["cv_rows"],
                result["cv_predictions"],
                display_name=result["display_name"],
                model_name=result["model_name"],
                model_family=result["model_family"],
                split_name="cv_oof",
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                result["test_rows"],
                result["test_predictions"],
                display_name=result["display_name"],
                model_name=result["model_name"],
                model_family=result["model_family"],
                split_name="test",
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=result["cv_rows"],
                predictions=result["cv_predictions"],
                model_name=result["display_name"],
                split_name="cv_oof",
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=result["test_rows"],
                predictions=result["test_predictions"],
                model_name=result["display_name"],
                split_name="test",
            )
        )
        true_distance_summary_rows.extend(
            _build_true_distance_summary(
                result["cv_rows"],
                result["cv_predictions"],
                display_name=result["display_name"],
                model_name=result["model_name"],
                split_name="cv_oof",
            )
        )
        true_distance_summary_rows.extend(
            _build_true_distance_summary(
                result["test_rows"],
                result["test_predictions"],
                display_name=result["display_name"],
                model_name=result["model_name"],
                split_name="test",
            )
        )
        if "fold_metrics" in result:
            fold_metric_rows.extend(
                _fold_summary_rows(
                    result["fold_metrics"],
                    display_name=result["display_name"],
                    model_name=result["model_name"],
                    split_name="cv_fold",
                )
            )
        if "coefficient_rows" in result:
            for row in result["coefficient_rows"]:
                coefficient_rows.append({"display_name": result["display_name"], **row})
        if "lower_feature_importance_rows" in result:
            lower_model_importance_rows.extend(
                [{"display_name": result["display_name"], **row} for row in result["lower_feature_importance_rows"]]
            )
        if "candidate_rows" in result:
            exact_rf_candidate_rows.extend(result["candidate_rows"])
        if result["model_name"] == "rf_winner_baseline":
            aggregated_rf_candidate_rows.extend(aggregated_family["rf_candidate_rows"])
        if result["model_name"] == "xgb_winner_baseline":
            xgb_candidate_rows.extend(aggregated_family["xgb_candidate_rows"])

    poster_test_rows = [row for row in metrics_rows if row["role"] == "poster" and row["split_name"] == "test"]
    ranked_poster_test_rows = sorted(
        poster_test_rows,
        key=lambda row: (
            float(row["mae"]),
            float(row["mean_relative_error"]),
            str(row["display_name"]),
        ),
    )
    for rank_index, row in enumerate(ranked_poster_test_rows, start=1):
        row["poster_rank"] = rank_index

    _write_csv_rows(reports_dir / "model_metrics.csv", metrics_rows)
    _write_csv_rows(reports_dir / "distance_range_metrics.csv", distance_range_metric_rows)
    _write_csv_rows(reports_dir / "predictions.csv", prediction_rows)
    _write_csv_rows(reports_dir / "grouped_error_summary.csv", grouped_summary_rows)
    _write_csv_rows(reports_dir / "true_distance_summary.csv", true_distance_summary_rows)
    _write_csv_rows(reports_dir / "cv_fold_metrics.csv", fold_metric_rows)
    _write_csv_rows(reports_dir / "model_coefficients.csv", coefficient_rows)
    _write_csv_rows(reports_dir / "lower_model_feature_importances.csv", lower_model_importance_rows)
    _write_csv_rows(reports_dir / "exact_rf_candidate_metrics.csv", exact_rf_candidate_rows)
    _write_csv_rows(reports_dir / "aggregated_rf_candidate_metrics.csv", aggregated_rf_candidate_rows)
    _write_csv_rows(reports_dir / "aggregated_xgb_candidate_metrics.csv", xgb_candidate_rows)
    _write_csv_rows(reports_dir / "aggregated_blend_search.csv", aggregated_family["blend_search_rows"])
    _write_csv_rows(reports_dir / "poster_test_ranking.csv", ranked_poster_test_rows)

    summary = {
        "study_name": str(config["study_name"]),
        "config_path": str(config_path.resolve()),
        "num_common_rows": len(exact_common_rows),
        "num_dev_rows": len(split_payload["dev_indices"]),
        "num_test_rows": len(split_payload["test_indices"]),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "aggregated_source_model_name": str(dict(config["benchmark"])["aggregated_source_model_name"]),
        "aggregated_feature_count": len(aggregated_feature_names),
        "poster_test_ranking": ranked_poster_test_rows,
        "best_aggregated_rf_candidate": str(aggregated_family["best_rf"]["candidate_name"]),
        "best_aggregated_xgb_candidate": str(aggregated_family["best_xgb"]["candidate_name"]),
        "best_blend_row": aggregated_family["best_blend_row"],
        "reports": {
            "model_catalog_csv": str(reports_dir / "model_catalog.csv"),
            "model_metrics_csv": str(reports_dir / "model_metrics.csv"),
            "distance_range_metrics_csv": str(reports_dir / "distance_range_metrics.csv"),
            "predictions_csv": str(reports_dir / "predictions.csv"),
            "grouped_error_summary_csv": str(reports_dir / "grouped_error_summary.csv"),
            "true_distance_summary_csv": str(reports_dir / "true_distance_summary.csv"),
            "poster_test_ranking_csv": str(reports_dir / "poster_test_ranking.csv"),
        },
    }
    _write_json(reports_dir / "summary.json", summary)

    print("[attempt8] Rerun finished")
    print("  common_rows: {}".format(len(exact_common_rows)))
    print("  summary_json: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    run_study(config_path=args.config)


if __name__ == "__main__":
    main()
