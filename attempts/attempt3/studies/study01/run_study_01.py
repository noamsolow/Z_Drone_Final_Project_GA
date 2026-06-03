"""
Study 01 for attempt3: compare raw-depth linear calibration against a stacked
variant that replaces the raw depth scalar with the Study 06 small-RF output.
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

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.evaluation import (
    build_grouped_error_summary,
    evaluate_predictions,
)
from pipeline.calibration.features import (
    CONTINUOUS_FEATURE_NAMES,
    read_regression_records_csv,
)
from pipeline.calibration.linear_models import (
    GEOMETRY_FEATURE_NAMES,
    fit_custom_linear_model,
)
from pipeline.calibration.splits import (
    make_distance_range_label,
    make_fixed_holdout_and_cv_splits,
    make_stratified_cv_folds,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BASE_CATEGORICAL_FEATURE_NAMES = ["weather", "time_of_day"]
STUDY06_NON_DEPTH_COLUMNS = {
    "image_path",
    "label_path",
    "stratum_key",
    "selection_rank_within_stratum",
    "true_distance_m",
    "weather",
    "time_of_day",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt3 Study 01 stacked linear comparison.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 01 YAML config.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _normalize_path(path_value: str) -> str:
    return str(Path(path_value).resolve())


def _read_csv_rows(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    materialized_rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in materialized_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized_rows:
            writer.writerow(row)


def _write_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _select_rows(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [rows[index] for index in indices]


def _load_study06_feature_rows(input_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    raw_rows = _read_csv_rows(input_path)
    if not raw_rows:
        raise ValueError("Study 06 fused feature table is empty: {}".format(input_path))

    fieldnames = list(raw_rows[0].keys())
    depth_feature_names = [name for name in fieldnames if name not in STUDY06_NON_DEPTH_COLUMNS]
    rows: List[Dict[str, Any]] = []
    for raw_row in raw_rows:
        parsed = dict(raw_row)
        parsed["selection_rank_within_stratum"] = int(parsed["selection_rank_within_stratum"])
        parsed["true_distance_m"] = float(parsed["true_distance_m"])
        for feature_name in depth_feature_names:
            parsed[feature_name] = float(parsed[feature_name])
        rows.append(parsed)
    return rows, depth_feature_names


def _parse_feature_name_list(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in str(raw_value).split("|") if item.strip()]


def _select_lower_model_feature_names(
    metrics_csv_path: Path,
    requested_model_name: Optional[str],
    available_feature_names: Sequence[str],
) -> Tuple[str, List[str], Dict[str, Any]]:
    metric_rows = _read_csv_rows(metrics_csv_path)
    if not metric_rows:
        raise ValueError("Study 06 metrics CSV is empty: {}".format(metrics_csv_path))

    for row in metric_rows:
        row["cv_mae"] = float(row["cv_mae"])
        row["cv_mean_relative_error"] = float(row["cv_mean_relative_error"])

    if requested_model_name is not None:
        matches = [row for row in metric_rows if row["model_name"] == requested_model_name]
        if not matches:
            raise ValueError(
                "Requested lower model '{}' not found in {}.".format(
                    requested_model_name,
                    metrics_csv_path,
                )
            )
        selected_row = matches[0]
    else:
        selected_row = sorted(
            metric_rows,
            key=lambda row: (
                float(row["cv_mae"]),
                float(row["cv_mean_relative_error"]),
                str(row["model_name"]),
            ),
        )[0]

    feature_names = _parse_feature_name_list(str(selected_row.get("feature_names", "")))
    if not feature_names:
        feature_names = list(available_feature_names)

    missing = [name for name in feature_names if name not in available_feature_names]
    if missing:
        raise ValueError(
            "Lower-model features missing from Study 06 fused table: {}".format(missing)
        )

    return str(selected_row["model_name"]), feature_names, selected_row


def _merge_feature_sources(
    attempt1_features_csv: Path,
    study06_fused_features_csv: Path,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    base_records = read_regression_records_csv(attempt1_features_csv)
    study06_rows, study06_feature_names = _load_study06_feature_rows(study06_fused_features_csv)
    study06_by_image = {_normalize_path(row["image_path"]): row for row in study06_rows}

    merged_rows: List[Dict[str, Any]] = []
    missing_study06_paths: List[str] = []
    for base_record in base_records:
        image_key = _normalize_path(base_record.image_path)
        study06_row = study06_by_image.get(image_key)
        if study06_row is None:
            missing_study06_paths.append(base_record.image_path)
            continue

        if abs(float(study06_row["true_distance_m"]) - float(base_record.true_distance_m)) > 1e-6:
            raise ValueError("Distance mismatch for image '{}'.".format(base_record.image_path))
        if str(study06_row["weather"]) != str(base_record.weather):
            raise ValueError("Weather mismatch for image '{}'.".format(base_record.image_path))
        if str(study06_row["time_of_day"]) != str(base_record.time_of_day):
            raise ValueError("Time-of-day mismatch for image '{}'.".format(base_record.image_path))

        merged_row = base_record.to_dict()
        merged_row["study06_stratum_key"] = str(study06_row["stratum_key"])
        merged_row["study06_selection_rank_within_stratum"] = int(
            study06_row["selection_rank_within_stratum"]
        )
        for feature_name in study06_feature_names:
            merged_row[feature_name] = float(study06_row[feature_name])
        merged_rows.append(merged_row)

    base_image_keys = {_normalize_path(record.image_path) for record in base_records}
    extra_study06_rows = [
        row["image_path"] for row in study06_rows if _normalize_path(row["image_path"]) not in base_image_keys
    ]
    if not merged_rows:
        raise ValueError(
            "No shared image paths were found between {} and {}.".format(
                attempt1_features_csv,
                study06_fused_features_csv,
            )
        )

    merged_rows = sorted(
        merged_rows,
        key=lambda row: (
            float(row["true_distance_m"]),
            str(row["weather"]),
            str(row["time_of_day"]),
            str(row["image_path"]),
        ),
    )
    summary = {
        "num_attempt1_rows": len(base_records),
        "num_study06_rows": len(study06_rows),
        "num_merged_rows": len(merged_rows),
        "num_attempt1_rows_missing_from_study06": len(missing_study06_paths),
        "num_extra_study06_rows": len(extra_study06_rows),
        "study06_depth_feature_count": len(study06_feature_names),
        "attempt1_rows_missing_from_study06_example_paths": missing_study06_paths[:10],
        "extra_study06_rows_example_paths": extra_study06_rows[:10],
        "attempt1_rows_missing_from_study06_paths": missing_study06_paths,
        "extra_study06_rows_paths": extra_study06_rows,
    }
    return merged_rows, study06_feature_names, summary


def _compute_feature_correlations(
    rows: Sequence[Dict[str, Any]],
    continuous_feature_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    output_rows: List[Dict[str, Any]] = []
    for feature_name in continuous_feature_names:
        feature_values = np.asarray([float(row[feature_name]) for row in rows], dtype=np.float64)
        if feature_values.size == 0 or np.std(feature_values) == 0.0 or np.std(true_values) == 0.0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(feature_values, true_values)[0, 1])
        output_rows.append(
            {
                "feature_name": feature_name,
                "pearson_correlation": correlation,
                "count": int(feature_values.size),
            }
        )
    return sorted(output_rows, key=lambda row: str(row["feature_name"]))


def _load_or_create_splits(
    rows: Sequence[Dict[str, Any]],
    split_dir: Path,
    holdout_fraction: float,
    num_folds: int,
    seed: int,
) -> Dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    test_split_path = split_dir / "test_split.json"
    cv_folds_path = split_dir / "cv_folds.json"

    if test_split_path.exists() and cv_folds_path.exists():
        test_split = json.loads(test_split_path.read_text(encoding="utf-8"))
        cv_folds = json.loads(cv_folds_path.read_text(encoding="utf-8"))
        return {
            "test_indices": test_split["test_indices"],
            "dev_indices": test_split["dev_indices"],
            "cv_folds": cv_folds["cv_folds"],
            "seed": test_split["seed"],
            "holdout_fraction": test_split["holdout_fraction"],
            "num_folds": cv_folds["num_folds"],
        }

    split_data = make_fixed_holdout_and_cv_splits(
        records=rows,
        holdout_fraction=holdout_fraction,
        num_folds=num_folds,
        seed=seed,
    )
    test_split_payload = {
        "seed": split_data["seed"],
        "holdout_fraction": split_data["holdout_fraction"],
        "test_indices": split_data["test_indices"],
        "dev_indices": split_data["dev_indices"],
        "strata_by_index": split_data["strata_by_index"],
    }
    cv_folds_payload = {
        "seed": split_data["seed"],
        "num_folds": split_data["num_folds"],
        "cv_folds": split_data["cv_folds"],
    }
    _write_json(test_split_payload, test_split_path)
    _write_json(cv_folds_payload, cv_folds_path)
    return split_data


def _load_random_forest_regressor():
    try:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("scikit-learn is required for attempt3 Study 01: {}".format(exc)) from exc


def _fit_random_forest_estimator(params: Dict[str, Any]):
    estimator_cls = _load_random_forest_regressor()
    return estimator_cls(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        max_features=params.get("max_features", "sqrt"),
        random_state=int(params.get("random_state", 0)),
        n_jobs=int(params.get("n_jobs", -1)),
    )


def _build_feature_matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float64)
    return np.asarray(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=np.float64,
    )


def _fit_lower_model_and_predict(
    train_rows: Sequence[Dict[str, Any]],
    predict_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    rf_params: Dict[str, Any],
) -> np.ndarray:
    if not predict_rows:
        return np.zeros(0, dtype=np.float64)
    estimator = _fit_random_forest_estimator(rf_params)
    x_train = _build_feature_matrix(train_rows, lower_feature_names)
    y_train = np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64)
    estimator.fit(x_train, y_train)
    x_predict = _build_feature_matrix(predict_rows, lower_feature_names)
    return estimator.predict(x_predict)


def _cross_fit_lower_model(
    train_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    rf_params: Dict[str, Any],
    num_folds: int,
    seed: int,
    verbose: bool = False,
    progress_label: str = "",
) -> Dict[str, Any]:
    if len(train_rows) < 2:
        raise ValueError("Need at least two rows to cross-fit the lower model.")

    inner_num_folds = max(2, min(int(num_folds), len(train_rows)))
    fold_payload = make_stratified_cv_folds(train_rows, num_folds=inner_num_folds, seed=seed)
    folds = fold_payload["cv_folds"]
    oof_predictions = np.zeros(len(train_rows), dtype=np.float64)

    for fold_index, fold in enumerate(folds, start=1):
        if verbose:
            print(
                "[attempt3-study01] Lower RF cross-fit {}/{} | {} | train_rows={} | val_rows={}".format(
                    fold_index,
                    len(folds),
                    progress_label or "lower-model",
                    len(fold["train_indices"]),
                    len(fold["val_indices"]),
                )
            )
        inner_train_rows = _select_rows(train_rows, fold["train_indices"])
        inner_val_rows = _select_rows(train_rows, fold["val_indices"])
        fold_predictions = _fit_lower_model_and_predict(
            train_rows=inner_train_rows,
            predict_rows=inner_val_rows,
            lower_feature_names=lower_feature_names,
            rf_params=rf_params,
        )
        for local_index, prediction in zip(fold["val_indices"], fold_predictions):
            oof_predictions[int(local_index)] = float(prediction)

    full_estimator = _fit_random_forest_estimator(rf_params)
    if verbose:
        print(
            "[attempt3-study01] Lower RF refit on full split | {} | rows={}".format(
                progress_label or "lower-model",
                len(train_rows),
            )
        )
    full_estimator.fit(
        _build_feature_matrix(train_rows, lower_feature_names),
        np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64),
    )

    importance_rows = [
        {
            "model_name": "study06_small_rf",
            "model_family": "random_forest",
            "term": feature_name,
            "importance_type": "feature_importance",
            "importance_value": float(importance_value),
        }
        for feature_name, importance_value in zip(
            lower_feature_names,
            getattr(full_estimator, "feature_importances_", np.zeros(len(lower_feature_names))),
        )
    ]
    return {
        "oof_predictions": oof_predictions,
        "full_estimator": full_estimator,
        "feature_importance_rows": importance_rows,
    }


def _attach_prediction_feature(
    rows: Sequence[Dict[str, Any]],
    feature_name: str,
    predictions: Sequence[float],
) -> List[Dict[str, Any]]:
    augmented_rows: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        augmented = dict(row)
        augmented[feature_name] = float(prediction)
        augmented_rows.append(augmented)
    return augmented_rows


def _prediction_rows(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    model_name: str,
    split_name: str,
    model_family: str = "linear_regression",
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        signed_error = float(prediction - true_distance)
        output_row = {
            "model_name": model_name,
            "model_family": model_family,
            "split_name": split_name,
            "image_path": row["image_path"],
            "label_path": row["label_path"],
            "weather": row["weather"],
            "time_of_day": row["time_of_day"],
            "distance_range": make_distance_range_label(true_distance),
            "true_distance_m": true_distance,
            "predicted_distance_m": float(prediction),
            "signed_error_m": signed_error,
            "absolute_error_m": abs(signed_error),
            "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
        }
        if "relative_depth" in row:
            output_row["relative_depth"] = float(row["relative_depth"])
        if "depth_lower_model_prediction" in row:
            output_row["depth_lower_model_prediction"] = float(row["depth_lower_model_prediction"])
        output_rows.append(output_row)
    return output_rows


def _build_model_specs(config: Dict[str, Any], lower_prediction_feature_name: str) -> List[Dict[str, Any]]:
    all_specs: Dict[str, Dict[str, Any]] = {
        "raw_depth_only_linear": {
            "model_name": "raw_depth_only_linear",
            "continuous_feature_names": ["relative_depth"],
            "categorical_feature_names": [],
            "uses_lower_model": False,
        },
        "geometry_metadata_linear": {
            "model_name": "geometry_metadata_linear",
            "continuous_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "raw_depth_geometry_metadata_linear": {
            "model_name": "raw_depth_geometry_metadata_linear",
            "continuous_feature_names": ["relative_depth"] + list(GEOMETRY_FEATURE_NAMES),
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "small_rf_depth_only_linear": {
            "model_name": "small_rf_depth_only_linear",
            "continuous_feature_names": [lower_prediction_feature_name],
            "categorical_feature_names": [],
            "uses_lower_model": True,
        },
        "stacked_rf_depth_geometry_metadata_linear": {
            "model_name": "stacked_rf_depth_geometry_metadata_linear",
            "continuous_feature_names": [lower_prediction_feature_name] + list(GEOMETRY_FEATURE_NAMES),
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": True,
        },
    }

    requested_names = list(config.get("modeling", {}).get("model_names", all_specs.keys()))
    missing = [name for name in requested_names if name not in all_specs]
    if missing:
        raise ValueError("Unsupported model names in config: {}".format(missing))
    return [all_specs[name] for name in requested_names]


def _cv_summary_from_fold_rows(fold_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not fold_rows:
        return {}
    metric_names = [
        "mae",
        "median_absolute_error",
        "mean_relative_error",
        "median_relative_error",
        "rmse",
        "r2",
        "mean_signed_error",
        "median_signed_error",
        "p90_absolute_error",
        "p95_absolute_error",
        "max_absolute_error",
        "within_5m_rate",
        "within_10m_rate",
        "within_20m_rate",
        "overprediction_rate",
        "underprediction_rate",
    ]
    summary = {
        "num_folds": len(fold_rows),
        "mean_count": float(np.mean([row["overall"]["count"] for row in fold_rows])),
    }
    for metric_name in metric_names:
        summary["mean_{}".format(metric_name)] = float(
            np.mean([row["overall"][metric_name] for row in fold_rows])
        )
    return summary


def run_study(config_path: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    resolved_config = load_config(config_path)
    output_root = _resolve_repo_path(str(resolved_config["output_root"]))
    features_dir = output_root / "features"
    reports_dir = output_root / "reports"
    split_dir = output_root / "splits"
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    sources = resolved_config["sources"]
    attempt1_features_csv = _resolve_repo_path(str(sources["attempt1_features_csv"]))
    study06_fused_features_csv = _resolve_repo_path(str(sources["study06_fused_features_csv"]))
    study06_metrics_csv = _resolve_repo_path(str(sources["study06_metrics_csv"]))

    merged_rows, study06_feature_names, merge_summary = _merge_feature_sources(
        attempt1_features_csv=attempt1_features_csv,
        study06_fused_features_csv=study06_fused_features_csv,
    )
    merged_features_csv = features_dir / "merged_feature_table.csv"
    _write_csv_rows(merged_rows, merged_features_csv)

    attempt1_missing_paths = list(merge_summary.pop("attempt1_rows_missing_from_study06_paths", []))
    study06_extra_paths = list(merge_summary.pop("extra_study06_rows_paths", []))
    _write_csv_rows(
        [{"image_path": path} for path in attempt1_missing_paths],
        features_dir / "attempt1_rows_missing_from_study06.csv",
    )
    _write_csv_rows(
        [{"image_path": path} for path in study06_extra_paths],
        features_dir / "study06_rows_missing_from_attempt1.csv",
    )

    lower_model_config = dict(resolved_config.get("lower_model", {}))
    lower_prediction_feature_name = str(
        lower_model_config.get("score_feature_name", "depth_lower_model_prediction")
    )
    selected_lower_model_name, selected_lower_feature_names, selected_lower_model_row = (
        _select_lower_model_feature_names(
            metrics_csv_path=study06_metrics_csv,
            requested_model_name=lower_model_config.get("selected_model_name"),
            available_feature_names=study06_feature_names,
        )
    )
    model_specs = _build_model_specs(resolved_config, lower_prediction_feature_name)

    feature_summary = dict(merge_summary)
    feature_summary.update(
        {
            "study_name": str(resolved_config.get("study_name", "study01")),
            "attempt1_features_csv": str(attempt1_features_csv),
            "study06_fused_features_csv": str(study06_fused_features_csv),
            "study06_metrics_csv": str(study06_metrics_csv),
            "merged_features_csv": str(merged_features_csv),
            "attempt1_rows_missing_from_study06_csv": str(
                features_dir / "attempt1_rows_missing_from_study06.csv"
            ),
            "study06_rows_missing_from_attempt1_csv": str(
                features_dir / "study06_rows_missing_from_attempt1.csv"
            ),
            "selected_lower_model_name": selected_lower_model_name,
            "selected_lower_feature_names": list(selected_lower_feature_names),
            "num_selected_lower_features": len(selected_lower_feature_names),
            "selected_lower_model_metrics_row": selected_lower_model_row,
            "requested_models": [spec["model_name"] for spec in model_specs],
        }
    )
    _write_json(feature_summary, features_dir / "summary.json")

    correlation_feature_names = list(CONTINUOUS_FEATURE_NAMES) + list(study06_feature_names)
    _write_csv_rows(
        _compute_feature_correlations(merged_rows, correlation_feature_names),
        reports_dir / "feature_correlations.csv",
    )

    split_config = dict(resolved_config.get("split", {}))
    split_data = _load_or_create_splits(
        rows=merged_rows,
        split_dir=split_dir,
        holdout_fraction=float(split_config.get("holdout_fraction", 0.15)),
        num_folds=int(split_config.get("num_folds", 5)),
        seed=int(split_config.get("seed", 0)),
    )
    dev_rows = _select_rows(merged_rows, split_data["dev_indices"])
    test_rows = _select_rows(merged_rows, split_data["test_indices"])

    if verbose:
        print(
            "[attempt3-study01] Stacked linear comparison\n"
            "  config: {}\n"
            "  merged_rows: {}\n"
            "  shared_rows_used: {}\n"
            "  attempt1_only_rows_dropped: {}\n"
            "  study06_only_rows_dropped: {}\n"
            "  selected_lower_model: {}\n"
            "  lower_feature_count: {}\n"
            "  requested_models: {}".format(
                Path(config_path or DEFAULT_CONFIG_PATH),
                len(merged_rows),
                len(merged_rows),
                merge_summary["num_attempt1_rows_missing_from_study06"],
                merge_summary["num_extra_study06_rows"],
                selected_lower_model_name,
                len(selected_lower_feature_names),
                ", ".join(spec["model_name"] for spec in model_specs),
            )
        )

    lower_rf_params = dict(lower_model_config.get("random_forest", {}))
    lower_inner_num_folds = int(lower_model_config.get("inner_num_folds", 5))
    lower_inner_seed = int(lower_model_config.get("inner_seed", 0))
    uses_lower_model = any(spec["uses_lower_model"] for spec in model_specs)

    cv_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study01")),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "models": {},
    }
    cv_prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []
    cv_ranking_rows: List[Dict[str, Any]] = []

    for model_index, model_spec in enumerate(model_specs, start=1):
        model_name = str(model_spec["model_name"])
        if verbose:
            print(
                "[attempt3-study01] CV evaluation {}/{} | {} | uses_lower_model={}".format(
                    model_index,
                    len(model_specs),
                    model_name,
                    bool(model_spec["uses_lower_model"]),
                )
            )

        fold_rows: List[Dict[str, Any]] = []
        dev_prediction_lookup: Dict[int, float] = {}
        lower_feature_lookup: Dict[int, float] = {}
        for outer_fold_index, fold in enumerate(split_data["cv_folds"], start=1):
            train_rows = _select_rows(merged_rows, fold["train_indices"])
            val_rows = _select_rows(merged_rows, fold["val_indices"])
            if verbose:
                print(
                    "[attempt3-study01] Outer CV fold {}/{} | {} | train_rows={} | val_rows={}".format(
                        outer_fold_index,
                        len(split_data["cv_folds"]),
                        model_name,
                        len(train_rows),
                        len(val_rows),
                    )
                )

            if model_spec["uses_lower_model"]:
                lower_cache = _cross_fit_lower_model(
                    train_rows=train_rows,
                    lower_feature_names=selected_lower_feature_names,
                    rf_params=lower_rf_params,
                    num_folds=lower_inner_num_folds,
                    seed=lower_inner_seed + int(fold["fold_index"]),
                    verbose=verbose,
                    progress_label="{} | outer_fold={}".format(model_name, outer_fold_index),
                )
                train_augmented = _attach_prediction_feature(
                    train_rows,
                    lower_prediction_feature_name,
                    lower_cache["oof_predictions"],
                )
                val_lower_predictions = lower_cache["full_estimator"].predict(
                    _build_feature_matrix(val_rows, selected_lower_feature_names)
                )
                val_augmented = _attach_prediction_feature(
                    val_rows,
                    lower_prediction_feature_name,
                    val_lower_predictions,
                )
                for global_index, lower_prediction in zip(fold["val_indices"], val_lower_predictions):
                    lower_feature_lookup[int(global_index)] = float(lower_prediction)
            else:
                train_augmented = list(train_rows)
                val_augmented = list(val_rows)

            model = fit_custom_linear_model(
                records=train_augmented,
                model_name=model_name,
                continuous_feature_names=model_spec["continuous_feature_names"],
                categorical_feature_names=model_spec["categorical_feature_names"],
            )
            fold_predictions = model.predict(val_augmented)
            fold_evaluation = evaluate_predictions(val_augmented, fold_predictions)
            if verbose:
                print(
                    "[attempt3-study01] Completed outer CV fold {}/{} | {} | fold_mae={:.4f}".format(
                        outer_fold_index,
                        len(split_data["cv_folds"]),
                        model_name,
                        float(fold_evaluation["overall"]["mae"]),
                    )
                )
            fold_rows.append(
                {
                    "fold_index": int(fold["fold_index"]),
                    "overall": fold_evaluation["overall"],
                    "per_distance_range": fold_evaluation["per_distance_range"],
                }
            )
            for global_index, prediction in zip(fold["val_indices"], fold_predictions):
                dev_prediction_lookup[int(global_index)] = float(prediction)

        if len(dev_prediction_lookup) != len(split_data["dev_indices"]):
            raise ValueError(
                "CV predictions for model '{}' do not cover the full dev split.".format(model_name)
            )

        dev_predictions = np.asarray(
            [dev_prediction_lookup[int(index)] for index in split_data["dev_indices"]],
            dtype=np.float64,
        )
        if model_spec["uses_lower_model"]:
            dev_lower_predictions = np.asarray(
                [lower_feature_lookup[int(index)] for index in split_data["dev_indices"]],
                dtype=np.float64,
            )
            dev_rows_for_outputs = _attach_prediction_feature(
                dev_rows,
                lower_prediction_feature_name,
                dev_lower_predictions,
            )
        else:
            dev_rows_for_outputs = list(dev_rows)
        cv_oof_evaluation = evaluate_predictions(dev_rows, dev_predictions)
        if verbose:
            print(
                "[attempt3-study01] Finished CV model {}/{} | {} | oof_mae={:.4f}".format(
                    model_index,
                    len(model_specs),
                    model_name,
                    float(cv_oof_evaluation["overall"]["mae"]),
                )
            )
        cv_metrics["models"][model_name] = {
            "fold_metrics": fold_rows,
            "summary": _cv_summary_from_fold_rows(fold_rows),
            "oof_overall": cv_oof_evaluation["overall"],
            "oof_per_distance_range": cv_oof_evaluation["per_distance_range"],
        }
        cv_ranking_rows.append(
            {
                "model_name": model_name,
                "mean_fold_mae": cv_metrics["models"][model_name]["summary"]["mean_mae"],
                "mean_fold_mean_relative_error": cv_metrics["models"][model_name]["summary"][
                    "mean_mean_relative_error"
                ],
                "oof_mae": cv_oof_evaluation["overall"]["mae"],
                "oof_mean_relative_error": cv_oof_evaluation["overall"]["mean_relative_error"],
            }
        )
        cv_prediction_rows.extend(
            _prediction_rows(
                rows=dev_rows_for_outputs,
                predictions=dev_predictions,
                model_name=model_name,
                split_name="cv_oof",
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=dev_rows_for_outputs,
                predictions=dev_predictions,
                model_name=model_name,
                split_name="cv_oof",
            )
        )

    ranked_models = sorted(
        cv_ranking_rows,
        key=lambda row: (
            float(row["mean_fold_mae"]),
            float(row["mean_fold_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for rank_index, row in enumerate(ranked_models, start=1):
        row["rank"] = rank_index
    winner_model_name = str(ranked_models[0]["model_name"])
    if verbose:
        print(
            "[attempt3-study01] CV winner selected | {} | mean_fold_mae={:.4f} | oof_mae={:.4f}".format(
                winner_model_name,
                float(ranked_models[0]["mean_fold_mae"]),
                float(ranked_models[0]["oof_mae"]),
            )
        )
    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["ranking"] = ranked_models
    _write_json(cv_metrics, reports_dir / "cv_metrics.json")

    test_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study01")),
        "winner_model_name": winner_model_name,
        "selected_lower_model_name": selected_lower_model_name,
        "models": {},
    }
    coefficient_rows: List[Dict[str, Any]] = []
    test_prediction_rows: List[Dict[str, Any]] = []
    lower_model_importance_rows: List[Dict[str, Any]] = []
    lower_model_test_predictions: Optional[np.ndarray] = None

    shared_dev_lower_cache: Optional[Dict[str, Any]] = None
    shared_dev_augmented_rows: Optional[List[Dict[str, Any]]] = None
    shared_test_augmented_rows: Optional[List[Dict[str, Any]]] = None
    if uses_lower_model:
        if verbose:
            print(
                "[attempt3-study01] Building shared lower-model feature for held-out test | dev_rows={} | test_rows={}".format(
                    len(dev_rows),
                    len(test_rows),
                )
            )
        shared_dev_lower_cache = _cross_fit_lower_model(
            train_rows=dev_rows,
            lower_feature_names=selected_lower_feature_names,
            rf_params=lower_rf_params,
            num_folds=lower_inner_num_folds,
            seed=lower_inner_seed + 1000,
            verbose=verbose,
            progress_label="heldout-test-prep",
        )
        shared_dev_augmented_rows = _attach_prediction_feature(
            dev_rows,
            lower_prediction_feature_name,
            shared_dev_lower_cache["oof_predictions"],
        )
        lower_model_test_predictions = shared_dev_lower_cache["full_estimator"].predict(
            _build_feature_matrix(test_rows, selected_lower_feature_names)
        )
        shared_test_augmented_rows = _attach_prediction_feature(
            test_rows,
            lower_prediction_feature_name,
            lower_model_test_predictions,
        )
        lower_model_importance_rows = list(shared_dev_lower_cache["feature_importance_rows"])

    for model_index, model_spec in enumerate(model_specs, start=1):
        model_name = str(model_spec["model_name"])
        if verbose:
            print(
                "[attempt3-study01] Held-out test fit {}/{} | {} | uses_lower_model={}".format(
                    model_index,
                    len(model_specs),
                    model_name,
                    bool(model_spec["uses_lower_model"]),
                )
            )

        if model_spec["uses_lower_model"]:
            assert shared_dev_augmented_rows is not None
            assert shared_test_augmented_rows is not None
            train_rows_for_model = shared_dev_augmented_rows
            test_rows_for_model = shared_test_augmented_rows
        else:
            train_rows_for_model = dev_rows
            test_rows_for_model = test_rows

        model = fit_custom_linear_model(
            records=train_rows_for_model,
            model_name=model_name,
            continuous_feature_names=model_spec["continuous_feature_names"],
            categorical_feature_names=model_spec["categorical_feature_names"],
        )
        predictions = model.predict(test_rows_for_model)
        evaluation = evaluate_predictions(test_rows_for_model, predictions)
        if verbose:
            print(
                "[attempt3-study01] Completed held-out test fit {}/{} | {} | test_mae={:.4f}".format(
                    model_index,
                    len(model_specs),
                    model_name,
                    float(evaluation["overall"]["mae"]),
                )
            )
        test_metrics["models"][model_name] = {
            "is_cv_winner": model_name == winner_model_name,
            "overall": evaluation["overall"],
            "per_distance_range": evaluation["per_distance_range"],
        }
        coefficient_rows.extend(model.coefficient_rows())
        test_prediction_rows.extend(
            _prediction_rows(
                rows=test_rows_for_model,
                predictions=predictions,
                model_name=model_name,
                split_name="test",
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=test_rows_for_model,
                predictions=predictions,
                model_name=model_name,
                split_name="test",
            )
        )

    _write_json(test_metrics, reports_dir / "test_metrics.json")
    _write_csv_rows(grouped_summary_rows, reports_dir / "grouped_error_summary.csv")
    _write_csv_rows(coefficient_rows, reports_dir / "model_coefficients.csv")
    _write_csv_rows(cv_prediction_rows, reports_dir / "cv_predictions.csv")
    _write_csv_rows(test_prediction_rows, reports_dir / "test_predictions.csv")
    _write_csv_rows(lower_model_importance_rows, reports_dir / "lower_model_feature_importances.csv")
    _write_json(
        {
            "selected_lower_model_name": selected_lower_model_name,
            "selected_lower_feature_names": list(selected_lower_feature_names),
            "num_selected_lower_features": len(selected_lower_feature_names),
            "random_forest_params": lower_rf_params,
            "study06_metrics_row": selected_lower_model_row,
        },
        reports_dir / "lower_model_summary.json",
    )

    report_summary = {
        "study_name": str(resolved_config.get("study_name", "study01")),
        "num_merged_rows": len(merged_rows),
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "num_selected_lower_features": len(selected_lower_feature_names),
        "winner_model_name": winner_model_name,
        "winner_cv_mean_fold_mae": float(ranked_models[0]["mean_fold_mae"]),
        "winner_cv_oof_mae": float(ranked_models[0]["oof_mae"]),
        "winner_test_mae": float(test_metrics["models"][winner_model_name]["overall"]["mae"]),
        "merged_features_csv": str(merged_features_csv),
        "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
        "test_metrics_json": str(reports_dir / "test_metrics.json"),
    }
    if lower_model_test_predictions is not None:
        lower_only_eval = evaluate_predictions(
            _attach_prediction_feature(test_rows, lower_prediction_feature_name, lower_model_test_predictions),
            lower_model_test_predictions,
        )
        report_summary["lower_model_test_mae"] = float(lower_only_eval["overall"]["mae"])

    _write_json(report_summary, reports_dir / "summary.json")
    if verbose:
        print(
            "[attempt3-study01] Study complete\n"
            "  winner_model: {}\n"
            "  winner_test_mae: {:.4f}\n"
            "  summary_json: {}".format(
                winner_model_name,
                float(report_summary["winner_test_mae"]),
                reports_dir / "summary.json",
            )
        )
    return report_summary


def main() -> None:
    args = parse_args()
    run_study(config_path=args.config, verbose=True)


if __name__ == "__main__":
    main()
