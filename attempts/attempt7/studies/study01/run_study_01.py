"""
Attempt 7 Study 01: aggregated-jitter RF/XGBoost ensemble on the Attempt 4 split.

This study takes the exact aggregated-jitter feature family that produced the
Attempt 3 Study 03 ~5.21 MAE RF, maps the Attempt 4 source-image split onto
that one-row-per-image representation, and then trains RF, XGBoost, and a blend
under that shared source-image partition.
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


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions
from pipeline.calibration.splits import make_distance_range_label


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
AGGREGATED_META_COLUMNS = {
    "image_path",
    "label_path",
    "stratum_key",
    "true_distance_m",
    "weather",
    "time_of_day",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt7 Study 01 aggregated-jitter ensemble.")
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


def _normalize_path(path_value: str) -> str:
    return str(Path(path_value).resolve())


def _select_rows(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [rows[int(index)] for index in indices]


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


def _load_feature_names(selection_csv: Path, source_model_name: str) -> List[str]:
    rows = _read_csv_rows(selection_csv)
    matching = [row for row in rows if str(row.get("model_name", "")) == source_model_name]
    if not matching:
        raise ValueError(
            "Could not find feature-selection row for '{}' in '{}'.".format(
                source_model_name,
                selection_csv,
            )
        )
    raw_value = str(matching[0]["feature_names"])
    feature_names = [item.strip() for item in raw_value.split("|") if item.strip()]
    if not feature_names:
        raise ValueError("Selected feature set for '{}' was empty.".format(source_model_name))
    return feature_names


def _load_aggregated_rows(feature_table_csv: Path, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    raw_rows = _read_csv_rows(feature_table_csv)
    if not raw_rows:
        raise ValueError("Aggregated feature table is empty: {}".format(feature_table_csv))

    one_hot_names = [name for name in feature_names if "=" in name]
    numeric_feature_names = [name for name in feature_names if "=" not in name]

    parsed_rows: List[Dict[str, Any]] = []
    for raw_row in raw_rows:
        row = dict(raw_row)
        row["source_group_id"] = _normalize_path(str(row["image_path"]))
        row["true_distance_m"] = float(row["true_distance_m"])
        for name in numeric_feature_names:
            row[name] = float(row[name])
        for name in one_hot_names:
            base_name, level = name.split("=", 1)
            row[name] = 1.0 if str(row.get(base_name, "")) == level else 0.0
        parsed_rows.append(row)

    missing = [name for name in feature_names if name not in parsed_rows[0]]
    if missing:
        raise ValueError("Aggregated rows are missing required features: {}".format(missing))
    return parsed_rows


def _load_expanded_rows(feature_table_csv: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(feature_table_csv)
    if not rows:
        raise ValueError("Expanded feature table is empty: {}".format(feature_table_csv))
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["source_group_id"] = _normalize_path(str(new_row["source_group_id"]))
        parsed.append(new_row)
    return parsed


def _map_attempt4_split_to_aggregated(
    aggregated_rows: Sequence[Dict[str, Any]],
    expanded_rows: Sequence[Dict[str, Any]],
    attempt4_test_split: Dict[str, Any],
    attempt4_cv_folds: Dict[str, Any],
) -> Dict[str, Any]:
    aggregated_index_by_source = {
        str(row["source_group_id"]): index for index, row in enumerate(aggregated_rows)
    }
    if len(aggregated_index_by_source) != len(aggregated_rows):
        raise ValueError("Aggregated rows do not have unique source_group_id values.")

    expanded_source_ids = [str(row["source_group_id"]) for row in expanded_rows]

    test_source_ids = sorted(
        {expanded_source_ids[int(index)] for index in attempt4_test_split["test_indices"]}
    )
    dev_source_ids = sorted(
        {expanded_source_ids[int(index)] for index in attempt4_test_split["dev_indices"]}
    )
    if len(test_source_ids) + len(dev_source_ids) != len(aggregated_rows):
        raise ValueError(
            "Mapped Attempt 4 source ids do not cover the aggregated table cleanly."
        )

    test_indices = sorted(aggregated_index_by_source[source_id] for source_id in test_source_ids)
    dev_indices = sorted(aggregated_index_by_source[source_id] for source_id in dev_source_ids)

    cv_folds: List[Dict[str, Any]] = []
    for fold in attempt4_cv_folds["cv_folds"]:
        val_source_ids = sorted(
            {expanded_source_ids[int(index)] for index in fold["val_indices"]}
        )
        train_source_ids = sorted(
            {expanded_source_ids[int(index)] for index in fold["train_indices"]}
        )
        cv_folds.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_indices": sorted(
                    aggregated_index_by_source[source_id] for source_id in train_source_ids
                ),
                "val_indices": sorted(
                    aggregated_index_by_source[source_id] for source_id in val_source_ids
                ),
            }
        )

    return {
        "seed": attempt4_test_split["seed"],
        "holdout_fraction": attempt4_test_split["holdout_fraction"],
        "mapped_from_attempt4": True,
        "num_total_rows": len(aggregated_rows),
        "num_dev_rows": len(dev_indices),
        "num_test_rows": len(test_indices),
        "num_unique_dev_sources": len(dev_source_ids),
        "num_unique_test_sources": len(test_source_ids),
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "cv_folds": cv_folds,
        "test_source_group_ids": test_source_ids,
        "dev_source_group_ids": dev_source_ids,
    }


def _matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def _target_values(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)


def _prediction_rows(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    *,
    model_name: str,
    model_family: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        pred = float(prediction)
        signed_error = pred - true_distance
        output.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "split_name": split_name,
                "image_path": str(row["image_path"]),
                "label_path": str(row["label_path"]),
                "source_group_id": str(row["source_group_id"]),
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
    return output


def _candidate_metric_row(
    *,
    model_family: str,
    candidate_name: str,
    feature_names: Sequence[str],
    metrics: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_family": model_family,
        "candidate_name": candidate_name,
        "num_features": len(feature_names),
        "feature_names": " | ".join(feature_names),
        "oof_mae": float(metrics["mae"]),
        "oof_mean_relative_error": float(metrics["mean_relative_error"]),
        "oof_rmse": float(metrics["rmse"]),
        "oof_r2": float(metrics["r2"]),
        **{"param_{}".format(key): value for key, value in params.items()},
    }


def _fit_model(model_family: str, params: Dict[str, Any]) -> Any:
    if model_family == "rf":
        return _fit_rf(params)
    if model_family == "xgb":
        return _fit_xgb(params)
    raise ValueError("Unsupported model_family '{}'.".format(model_family))


def _evaluate_candidate_oof(
    *,
    model_family: str,
    candidate_name: str,
    params: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    cv_folds: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    oof_predictions = np.zeros(len(rows), dtype=np.float64)
    filled = np.zeros(len(rows), dtype=bool)

    for fold in cv_folds:
        train_rows = _select_rows(rows, fold["train_indices"])
        val_rows = _select_rows(rows, fold["val_indices"])
        estimator = _fit_model(model_family, params)
        estimator.fit(_matrix(train_rows, feature_names), _target_values(train_rows))
        fold_predictions = np.asarray(
            estimator.predict(_matrix(val_rows, feature_names)),
            dtype=np.float64,
        )
        for local_index, prediction in zip(fold["val_indices"], fold_predictions):
            oof_predictions[int(local_index)] = float(prediction)
            filled[int(local_index)] = True

    if not np.all(filled):
        raise ValueError(
            "OOF predictions were not fully populated for {} / {}.".format(
                model_family,
                candidate_name,
            )
        )

    metrics = evaluate_predictions(rows, oof_predictions)["overall"]
    return {
        "model_family": model_family,
        "candidate_name": candidate_name,
        "params": dict(params),
        "feature_names": list(feature_names),
        "oof_predictions": oof_predictions,
        "oof_metrics": dict(metrics),
        "oof_prediction_rows": _prediction_rows(
            rows,
            oof_predictions,
            model_name="{}__{}".format(model_family, candidate_name),
            model_family=model_family,
            split_name="cv_oof",
        ),
    }


def _fit_selected_on_test(
    *,
    model_family: str,
    candidate_name: str,
    params: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    estimator = _fit_model(model_family, params)
    estimator.fit(_matrix(train_rows, feature_names), _target_values(train_rows))
    predictions = np.asarray(estimator.predict(_matrix(test_rows, feature_names)), dtype=np.float64)
    metrics_payload = evaluate_predictions(test_rows, predictions)
    return {
        "model_family": model_family,
        "candidate_name": candidate_name,
        "predictions": predictions,
        "metrics": metrics_payload,
        "prediction_rows": _prediction_rows(
            test_rows,
            predictions,
            model_name="{}__{}".format(model_family, candidate_name),
            model_family=model_family,
            split_name="test",
        ),
    }


def _select_best_candidate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        results,
        key=lambda row: (
            float(row["oof_metrics"]["mae"]),
            float(row["oof_metrics"]["mean_relative_error"]),
            str(row["candidate_name"]),
        ),
    )[0]


def _blend_predictions(
    rf_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
    *,
    rf_weight: float,
    clip_min_distance_m: float,
) -> np.ndarray:
    blended = (rf_weight * rf_predictions) + ((1.0 - rf_weight) * xgb_predictions)
    if clip_min_distance_m is not None:
        blended = np.maximum(blended, float(clip_min_distance_m))
    return blended


def _blend_search(
    *,
    rows: Sequence[Dict[str, Any]],
    rf_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
    weight_grid_step: float,
    clip_min_distance_m: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if weight_grid_step <= 0.0 or weight_grid_step > 1.0:
        raise ValueError("weight_grid_step must be in (0, 1].")
    steps = int(round(1.0 / weight_grid_step))
    candidate_weights = sorted({round(index * weight_grid_step, 10) for index in range(steps + 1)} | {1.0})
    search_rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None

    for rf_weight in candidate_weights:
        predictions = _blend_predictions(
            rf_predictions,
            xgb_predictions,
            rf_weight=float(rf_weight),
            clip_min_distance_m=clip_min_distance_m,
        )
        overall = evaluate_predictions(rows, predictions)["overall"]
        row = {
            "rf_weight": float(rf_weight),
            "xgb_weight": float(1.0 - rf_weight),
            "oof_mae": float(overall["mae"]),
            "oof_mean_relative_error": float(overall["mean_relative_error"]),
            "oof_rmse": float(overall["rmse"]),
            "oof_r2": float(overall["r2"]),
        }
        search_rows.append(row)
        if best_row is None or (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            float(row["rf_weight"]),
        ) < (
            float(best_row["oof_mae"]),
            float(best_row["oof_mean_relative_error"]),
            float(best_row["rf_weight"]),
        ):
            best_row = row

    if best_row is None:
        raise ValueError("Blend search did not produce any candidate rows.")
    return search_rows, best_row


def _blend_prediction_rows(
    rows: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    *,
    model_name: str,
    split_name: str,
    rf_weight: float,
    xgb_weight: float,
    rf_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
) -> List[Dict[str, Any]]:
    base_rows = _prediction_rows(
        rows,
        predictions,
        model_name=model_name,
        model_family="blend",
        split_name=split_name,
    )
    for row, rf_pred, xgb_pred in zip(base_rows, rf_predictions, xgb_predictions):
        row["rf_prediction_m"] = float(rf_pred)
        row["xgb_prediction_m"] = float(xgb_pred)
        row["rf_weight"] = float(rf_weight)
        row["xgb_weight"] = float(xgb_weight)
    return base_rows


def run_study(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _read_yaml(config_path)
    sources = dict(config["sources"])
    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    splits_dir = output_root / "splits"
    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    feature_names = _load_feature_names(
        _resolve_repo_path(str(sources["aggregated_feature_selection_csv"])),
        str(dict(config["features"])["source_model_name"]),
    )
    aggregated_rows = _load_aggregated_rows(
        _resolve_repo_path(str(sources["aggregated_feature_table_csv"])),
        feature_names,
    )
    expanded_rows = _load_expanded_rows(
        _resolve_repo_path(str(sources["attempt4_expanded_feature_table_csv"]))
    )
    mapped_split = _map_attempt4_split_to_aggregated(
        aggregated_rows=aggregated_rows,
        expanded_rows=expanded_rows,
        attempt4_test_split=_read_json(_resolve_repo_path(str(sources["attempt4_test_split_json"]))),
        attempt4_cv_folds=_read_json(_resolve_repo_path(str(sources["attempt4_cv_folds_json"]))),
    )
    _write_json(splits_dir / "mapped_attempt4_split.json", mapped_split)

    dev_rows = _select_rows(aggregated_rows, mapped_split["dev_indices"])
    test_rows = _select_rows(aggregated_rows, mapped_split["test_indices"])
    cv_folds = [
        {
            "fold_index": int(fold["fold_index"]),
            "train_indices": [mapped_split["dev_indices"].index(index) for index in fold["train_indices"]],
            "val_indices": [mapped_split["dev_indices"].index(index) for index in fold["val_indices"]],
        }
        for fold in mapped_split["cv_folds"]
    ]

    rf_results: List[Dict[str, Any]] = []
    for candidate in list(dict(config["models"])["rf_candidates"]):
        candidate = dict(candidate)
        name = str(candidate.pop("name"))
        rf_results.append(
            _evaluate_candidate_oof(
                model_family="rf",
                candidate_name=name,
                params=candidate,
                rows=dev_rows,
                feature_names=feature_names,
                cv_folds=cv_folds,
            )
        )
        print(
            "[attempt7-study01] RF candidate {} | oof_mae={:.4f}".format(
                name,
                float(rf_results[-1]["oof_metrics"]["mae"]),
            )
        )

    xgb_results: List[Dict[str, Any]] = []
    for candidate in list(dict(config["models"])["xgb_candidates"]):
        candidate = dict(candidate)
        name = str(candidate.pop("name"))
        xgb_results.append(
            _evaluate_candidate_oof(
                model_family="xgb",
                candidate_name=name,
                params=candidate,
                rows=dev_rows,
                feature_names=feature_names,
                cv_folds=cv_folds,
            )
        )
        print(
            "[attempt7-study01] XGB candidate {} | oof_mae={:.4f}".format(
                name,
                float(xgb_results[-1]["oof_metrics"]["mae"]),
            )
        )

    best_rf = _select_best_candidate(rf_results)
    best_xgb = _select_best_candidate(xgb_results)
    print(
        "[attempt7-study01] Selected RF={} | XGB={}".format(
            best_rf["candidate_name"],
            best_xgb["candidate_name"],
        )
    )

    rf_test = _fit_selected_on_test(
        model_family="rf",
        candidate_name=str(best_rf["candidate_name"]),
        params=dict(best_rf["params"]),
        train_rows=dev_rows,
        test_rows=test_rows,
        feature_names=feature_names,
    )
    xgb_test = _fit_selected_on_test(
        model_family="xgb",
        candidate_name=str(best_xgb["candidate_name"]),
        params=dict(best_xgb["params"]),
        train_rows=dev_rows,
        test_rows=test_rows,
        feature_names=feature_names,
    )

    blend_config = dict(config["blend"])
    blend_search_rows, best_blend_row = _blend_search(
        rows=dev_rows,
        rf_predictions=np.asarray(best_rf["oof_predictions"], dtype=np.float64),
        xgb_predictions=np.asarray(best_xgb["oof_predictions"], dtype=np.float64),
        weight_grid_step=float(blend_config["weight_grid_step"]),
        clip_min_distance_m=float(blend_config.get("clip_min_distance_m", 0.0)),
    )

    model_specs = [
        {"model_name": "rf_winner_baseline", "rf_weight": 1.0, "xgb_weight": 0.0},
        {"model_name": "xgb_winner_baseline", "rf_weight": 0.0, "xgb_weight": 1.0},
        {"model_name": "equal_weight_blend", "rf_weight": 0.5, "xgb_weight": 0.5},
        {
            "model_name": "tuned_weight_blend",
            "rf_weight": float(best_blend_row["rf_weight"]),
            "xgb_weight": float(best_blend_row["xgb_weight"]),
        },
    ]

    cv_prediction_rows: List[Dict[str, Any]] = []
    test_prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []
    cv_metrics: Dict[str, Any] = {"models": {}}
    test_metrics: Dict[str, Any] = {"models": {}}

    for spec in model_specs:
        model_name = str(spec["model_name"])
        rf_weight = float(spec["rf_weight"])
        xgb_weight = float(spec["xgb_weight"])
        cv_predictions = _blend_predictions(
            np.asarray(best_rf["oof_predictions"], dtype=np.float64),
            np.asarray(best_xgb["oof_predictions"], dtype=np.float64),
            rf_weight=rf_weight,
            clip_min_distance_m=float(blend_config.get("clip_min_distance_m", 0.0)),
        )
        test_predictions = _blend_predictions(
            np.asarray(rf_test["predictions"], dtype=np.float64),
            np.asarray(xgb_test["predictions"], dtype=np.float64),
            rf_weight=rf_weight,
            clip_min_distance_m=float(blend_config.get("clip_min_distance_m", 0.0)),
        )
        cv_eval = evaluate_predictions(dev_rows, cv_predictions)
        test_eval = evaluate_predictions(test_rows, test_predictions)
        cv_metrics["models"][model_name] = {
            "rf_weight": rf_weight,
            "xgb_weight": xgb_weight,
            "overall": dict(cv_eval["overall"]),
            "per_distance_range": dict(cv_eval["per_distance_range"]),
        }
        test_metrics["models"][model_name] = {
            "rf_weight": rf_weight,
            "xgb_weight": xgb_weight,
            "overall": dict(test_eval["overall"]),
            "per_distance_range": dict(test_eval["per_distance_range"]),
        }
        cv_prediction_rows.extend(
            _blend_prediction_rows(
                dev_rows,
                cv_predictions,
                model_name=model_name,
                split_name="cv_oof",
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
                rf_predictions=np.asarray(best_rf["oof_predictions"], dtype=np.float64),
                xgb_predictions=np.asarray(best_xgb["oof_predictions"], dtype=np.float64),
            )
        )
        test_prediction_rows.extend(
            _blend_prediction_rows(
                test_rows,
                test_predictions,
                model_name=model_name,
                split_name="test",
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
                rf_predictions=np.asarray(rf_test["predictions"], dtype=np.float64),
                xgb_predictions=np.asarray(xgb_test["predictions"], dtype=np.float64),
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=dev_rows,
                predictions=cv_predictions,
                model_name=model_name,
                split_name="cv_oof",
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=test_rows,
                predictions=test_predictions,
                model_name=model_name,
                split_name="test",
            )
        )
        print(
            "[attempt7-study01] {} | cv_mae={:.4f} | test_mae={:.4f}".format(
                model_name,
                float(cv_eval["overall"]["mae"]),
                float(test_eval["overall"]["mae"]),
            )
        )

    rf_candidate_rows = [
        _candidate_metric_row(
            model_family="rf",
            candidate_name=str(result["candidate_name"]),
            feature_names=feature_names,
            metrics=dict(result["oof_metrics"]),
            params=dict(result["params"]),
        )
        for result in rf_results
    ]
    xgb_candidate_rows = [
        _candidate_metric_row(
            model_family="xgb",
            candidate_name=str(result["candidate_name"]),
            feature_names=feature_names,
            metrics=dict(result["oof_metrics"]),
            params=dict(result["params"]),
        )
        for result in xgb_results
    ]

    ranking_rows = []
    for name, payload in cv_metrics["models"].items():
        ranking_rows.append(
            {
                "model_name": name,
                "oof_mae": float(payload["overall"]["mae"]),
                "oof_mean_relative_error": float(payload["overall"]["mean_relative_error"]),
            }
        )
    ranked_models = sorted(
        ranking_rows,
        key=lambda row: (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for index, row in enumerate(ranked_models, start=1):
        row["rank"] = index
    winner_model_name = str(ranked_models[0]["model_name"])

    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["rf_winner_candidate_name"] = str(best_rf["candidate_name"])
    cv_metrics["xgb_winner_candidate_name"] = str(best_xgb["candidate_name"])
    cv_metrics["blend_search_best_row"] = dict(best_blend_row)
    cv_metrics["ranking"] = ranked_models
    test_metrics["winner_model_name"] = winner_model_name
    test_metrics["rf_winner_candidate_name"] = str(best_rf["candidate_name"])
    test_metrics["xgb_winner_candidate_name"] = str(best_xgb["candidate_name"])

    _write_csv_rows(reports_dir / "rf_candidate_metrics.csv", rf_candidate_rows)
    _write_csv_rows(reports_dir / "xgb_candidate_metrics.csv", xgb_candidate_rows)
    _write_csv_rows(reports_dir / "blend_search.csv", blend_search_rows)
    _write_csv_rows(reports_dir / "cv_predictions.csv", cv_prediction_rows)
    _write_csv_rows(reports_dir / "test_predictions.csv", test_prediction_rows)
    _write_csv_rows(reports_dir / "grouped_error_summary.csv", grouped_summary_rows)
    _write_json(reports_dir / "cv_metrics.json", cv_metrics)
    _write_json(reports_dir / "test_metrics.json", test_metrics)

    attempt4_rf_summary = _read_json(_resolve_repo_path(str(sources["attempt4_rf_summary_json"])))
    attempt4_xgb_summary = _read_json(_resolve_repo_path(str(sources["attempt4_xgb_summary_json"])))
    attempt4_blend_summary = _read_json(_resolve_repo_path(str(sources["attempt4_blend_summary_json"])))
    attempt4_blend_test_metrics = _read_json(_resolve_repo_path(str(sources["attempt4_blend_test_metrics_json"])))

    attempt4_equal_weight_mae = float(
        attempt4_blend_test_metrics["models"]["equal_weight_blend"]["overall"]["mae"]
    )
    attempt4_tuned_weight_mae = float(
        attempt4_blend_test_metrics["models"]["tuned_weight_blend"]["overall"]["mae"]
    )
    attempt7_test_winner = dict(test_metrics["models"][winner_model_name]["overall"])

    comparison_rows = [
        {
            "benchmark_name": "attempt4_rf_baseline",
            "test_mae": float(attempt4_rf_summary["winner_test_mae"]),
            "test_mean_relative_error": float(attempt4_rf_summary["winner_test_mean_relative_error"]),
        },
        {
            "benchmark_name": "attempt4_xgb_baseline",
            "test_mae": float(attempt4_xgb_summary["winner_test_mae"]),
            "test_mean_relative_error": float(attempt4_xgb_summary["winner_test_mean_relative_error"]),
        },
        {
            "benchmark_name": "attempt4_equal_weight_blend",
            "test_mae": attempt4_equal_weight_mae,
            "test_mean_relative_error": float(
                attempt4_blend_test_metrics["models"]["equal_weight_blend"]["overall"]["mean_relative_error"]
            ),
        },
        {
            "benchmark_name": "attempt4_tuned_weight_blend",
            "test_mae": attempt4_tuned_weight_mae,
            "test_mean_relative_error": float(attempt4_blend_summary["winner_test_mean_relative_error"]),
        },
        {
            "benchmark_name": "attempt7_winner",
            "test_mae": float(attempt7_test_winner["mae"]),
            "test_mean_relative_error": float(attempt7_test_winner["mean_relative_error"]),
        },
    ]
    _write_csv_rows(reports_dir / "comparison_to_attempt4.csv", comparison_rows)

    summary = {
        "study_name": str(config["study_name"]),
        "config_path": str(config_path.resolve()),
        "aggregated_feature_table_csv": str(
            _resolve_repo_path(str(sources["aggregated_feature_table_csv"]))
        ),
        "selected_source_model_name": str(dict(config["features"])["source_model_name"]),
        "selected_feature_names": list(feature_names),
        "num_selected_features": len(feature_names),
        "num_total_rows": len(aggregated_rows),
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "mapped_from_attempt4_source_split": True,
        "rf_winner_candidate_name": str(best_rf["candidate_name"]),
        "xgb_winner_candidate_name": str(best_xgb["candidate_name"]),
        "winner_model_name": winner_model_name,
        "winner_cv_oof_mae": float(ranked_models[0]["oof_mae"]),
        "winner_cv_oof_mean_relative_error": float(ranked_models[0]["oof_mean_relative_error"]),
        "winner_test_mae": float(attempt7_test_winner["mae"]),
        "winner_test_mean_relative_error": float(attempt7_test_winner["mean_relative_error"]),
        "best_blend_search_row": dict(best_blend_row),
        "attempt4_comparison": {
            "attempt4_rf_test_mae": float(attempt4_rf_summary["winner_test_mae"]),
            "attempt4_xgb_test_mae": float(attempt4_xgb_summary["winner_test_mae"]),
            "attempt4_equal_weight_blend_test_mae": attempt4_equal_weight_mae,
            "attempt4_tuned_weight_blend_test_mae": attempt4_tuned_weight_mae,
            "attempt7_winner_test_mae": float(attempt7_test_winner["mae"]),
            "delta_vs_attempt4_tuned_blend_mae": float(attempt7_test_winner["mae"]) - attempt4_tuned_weight_mae,
            "delta_vs_attempt4_equal_blend_mae": float(attempt7_test_winner["mae"]) - attempt4_equal_weight_mae,
        },
        "reports": {
            "rf_candidate_metrics_csv": str(reports_dir / "rf_candidate_metrics.csv"),
            "xgb_candidate_metrics_csv": str(reports_dir / "xgb_candidate_metrics.csv"),
            "blend_search_csv": str(reports_dir / "blend_search.csv"),
            "comparison_to_attempt4_csv": str(reports_dir / "comparison_to_attempt4.csv"),
            "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
            "test_metrics_json": str(reports_dir / "test_metrics.json"),
        },
    }
    _write_json(reports_dir / "summary.json", summary)

    analysis_lines = [
        "# Attempt 7 Study 01",
        "",
        "This study reuses the exact aggregated-jitter feature family from the Attempt 3 Study 03 `raw_depth_geometry_metadata_rf` winner,",
        "but remaps the Attempt 4 source-image split onto the one-row-per-image aggregated table so the RF/XGB/blend comparison stays as fair as possible.",
        "",
        "## Setup",
        "",
        "- Feature source model: `{}`".format(dict(config["features"])["source_model_name"]),
        "- Number of selected features: `{}`".format(len(feature_names)),
        "- Dev rows: `{}`".format(len(dev_rows)),
        "- Test rows: `{}`".format(len(test_rows)),
        "- RF winner candidate: `{}`".format(best_rf["candidate_name"]),
        "- XGB winner candidate: `{}`".format(best_xgb["candidate_name"]),
        "",
        "## Main Result",
        "",
        "- Attempt 7 winner: `{}`".format(winner_model_name),
        "- Attempt 7 winner test MAE: `{:.4f}`".format(float(attempt7_test_winner["mae"])),
        "- Attempt 7 winner test mean relative error: `{:.4f}`".format(float(attempt7_test_winner["mean_relative_error"])),
        "",
        "## Comparison To Attempt 4",
        "",
        "- Attempt 4 tuned blend test MAE: `{:.4f}`".format(attempt4_tuned_weight_mae),
        "- Attempt 4 equal-weight blend test MAE: `{:.4f}`".format(attempt4_equal_weight_mae),
        "- Attempt 7 delta vs Attempt 4 tuned blend: `{:+.4f}`".format(float(attempt7_test_winner["mae"]) - attempt4_tuned_weight_mae),
        "- Attempt 7 delta vs Attempt 4 equal blend: `{:+.4f}`".format(float(attempt7_test_winner["mae"]) - attempt4_equal_weight_mae),
    ]
    (reports_dir / "analysis.md").write_text("\n".join(analysis_lines), encoding="utf-8")

    print("[attempt7-study01] Study complete")
    print("  winner_model: {}".format(winner_model_name))
    print("  winner_test_mae: {:.4f}".format(float(attempt7_test_winner["mae"])))
    print("  summary_json: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    run_study(config_path=args.config)


if __name__ == "__main__":
    main()
