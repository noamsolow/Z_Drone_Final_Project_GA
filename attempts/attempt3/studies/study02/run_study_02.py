"""
Study 02 for attempt3: replace the stage-two linear model with a larger random
forest, while keeping the small Study 06 random forest as a leakage-safe
stacked depth feature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study01.run_study_01 import (
    BASE_CATEGORICAL_FEATURE_NAMES,
    _attach_prediction_feature,
    _compute_feature_correlations,
    _cross_fit_lower_model,
    _cv_summary_from_fold_rows,
    _fit_random_forest_estimator,
    _load_or_create_splits,
    _merge_feature_sources,
    _prediction_rows,
    _resolve_repo_path,
    _select_lower_model_feature_names,
    _select_rows,
    _write_csv_rows,
    _write_json,
)
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions
from pipeline.calibration.features import CONTINUOUS_FEATURE_NAMES
from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt3 Study 02 stacked big-random-forest comparison.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 02 YAML config.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_global_categorical_levels(
    rows: Sequence[Dict[str, Any]],
    categorical_feature_names: Sequence[str],
) -> Dict[str, List[str]]:
    levels: Dict[str, List[str]] = {}
    for feature_name in categorical_feature_names:
        levels[feature_name] = sorted({str(row[feature_name]) for row in rows})
    return levels


def _build_rf_feature_names(
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Dict[str, Sequence[str]],
) -> List[str]:
    feature_names = list(continuous_feature_names)
    for feature_name in categorical_feature_names:
        for level in categorical_levels.get(feature_name, []):
            feature_names.append("{}={}".format(feature_name, level))
    return feature_names


def _build_rf_matrix(
    rows: Sequence[Dict[str, Any]],
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Dict[str, Sequence[str]],
) -> np.ndarray:
    matrix_rows: List[List[float]] = []
    for row in rows:
        values: List[float] = [float(row[name]) for name in continuous_feature_names]
        for feature_name in categorical_feature_names:
            row_value = str(row[feature_name])
            for level in categorical_levels.get(feature_name, []):
                values.append(1.0 if row_value == level else 0.0)
        matrix_rows.append(values)
    if not matrix_rows:
        return np.zeros(
            (
                0,
                len(_build_rf_feature_names(continuous_feature_names, categorical_feature_names, categorical_levels)),
            ),
            dtype=np.float64,
        )
    return np.asarray(matrix_rows, dtype=np.float64)


def _candidate_params_by_name(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    candidate_rows = list(config.get("upper_random_forest", {}).get("candidate_configs", []))
    if not candidate_rows:
        raise ValueError("upper_random_forest.candidate_configs must not be empty.")
    output: Dict[str, Dict[str, Any]] = {}
    for row in candidate_rows:
        name = str(row["name"])
        if name in output:
            raise ValueError("Duplicate upper RF candidate name '{}'.".format(name))
        params = dict(row)
        params.pop("name", None)
        output[name] = params
    return output


def _build_model_specs(config: Dict[str, Any], lower_prediction_feature_name: str) -> List[Dict[str, Any]]:
    candidate_name_map = {
        str(key): [str(value) for value in values]
        for key, values in dict(config.get("upper_random_forest", {}).get("model_candidate_names", {})).items()
    }
    all_specs: Dict[str, Dict[str, Any]] = {
        "raw_depth_only_rf": {
            "model_name": "raw_depth_only_rf",
            "continuous_feature_names": ["relative_depth"],
            "categorical_feature_names": [],
            "uses_lower_model": False,
        },
        "geometry_metadata_rf": {
            "model_name": "geometry_metadata_rf",
            "continuous_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "raw_depth_geometry_metadata_rf": {
            "model_name": "raw_depth_geometry_metadata_rf",
            "continuous_feature_names": ["relative_depth"] + list(GEOMETRY_FEATURE_NAMES),
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "small_rf_depth_only_rf": {
            "model_name": "small_rf_depth_only_rf",
            "continuous_feature_names": [lower_prediction_feature_name],
            "categorical_feature_names": [],
            "uses_lower_model": True,
        },
        "stacked_rf_depth_geometry_metadata_rf": {
            "model_name": "stacked_rf_depth_geometry_metadata_rf",
            "continuous_feature_names": [lower_prediction_feature_name] + list(GEOMETRY_FEATURE_NAMES),
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": True,
        },
    }
    requested_names = list(config.get("modeling", {}).get("model_names", all_specs.keys()))
    missing = [name for name in requested_names if name not in all_specs]
    if missing:
        raise ValueError("Unsupported model names in config: {}".format(missing))
    model_specs: List[Dict[str, Any]] = []
    for name in requested_names:
        spec = dict(all_specs[name])
        spec["candidate_names"] = list(candidate_name_map.get(name, []))
        model_specs.append(spec)
    return model_specs


def _prepare_outer_fold_caches(
    merged_rows: Sequence[Dict[str, Any]],
    cv_folds: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    lower_prediction_feature_name: str,
    lower_rf_params: Dict[str, Any],
    lower_inner_num_folds: int,
    lower_inner_seed: int,
    needs_lower_model: bool,
    verbose: bool,
) -> Dict[int, Dict[str, Any]]:
    caches: Dict[int, Dict[str, Any]] = {}
    for outer_fold_index, fold in enumerate(cv_folds, start=1):
        train_rows = _select_rows(merged_rows, fold["train_indices"])
        val_rows = _select_rows(merged_rows, fold["val_indices"])
        cache: Dict[str, Any] = {
            "fold_index": int(fold["fold_index"]),
            "train_base_rows": train_rows,
            "val_base_rows": val_rows,
            "global_val_indices": [int(index) for index in fold["val_indices"]],
        }
        if needs_lower_model:
            if verbose:
                print(
                    "[attempt3-study02] Preparing lower-model cache for outer fold {}/{} | train_rows={} | val_rows={}".format(
                        outer_fold_index,
                        len(cv_folds),
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
                verbose=verbose,
                progress_label="study02 outer_fold={}".format(outer_fold_index),
            )
            train_augmented = _attach_prediction_feature(
                train_rows,
                lower_prediction_feature_name,
                lower_cache["oof_predictions"],
            )
            val_lower_predictions = lower_cache["full_estimator"].predict(
                np.asarray(
                    [[float(row[name]) for name in lower_feature_names] for row in val_rows],
                    dtype=np.float64,
                )
            )
            val_augmented = _attach_prediction_feature(
                val_rows,
                lower_prediction_feature_name,
                val_lower_predictions,
            )
            cache["train_lower_rows"] = train_augmented
            cache["val_lower_rows"] = val_augmented
        caches[int(fold["fold_index"])] = cache
    return caches


def _prepare_heldout_lower_cache(
    dev_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    lower_prediction_feature_name: str,
    lower_rf_params: Dict[str, Any],
    lower_inner_num_folds: int,
    lower_inner_seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    if verbose:
        print(
            "[attempt3-study02] Building shared lower-model feature for held-out test | dev_rows={} | test_rows={}".format(
                len(dev_rows),
                len(test_rows),
            )
        )
    lower_cache = _cross_fit_lower_model(
        train_rows=dev_rows,
        lower_feature_names=lower_feature_names,
        rf_params=lower_rf_params,
        num_folds=lower_inner_num_folds,
        seed=lower_inner_seed + 1000,
        verbose=verbose,
        progress_label="study02 heldout-test-prep",
    )
    dev_augmented = _attach_prediction_feature(
        dev_rows,
        lower_prediction_feature_name,
        lower_cache["oof_predictions"],
    )
    test_lower_predictions = lower_cache["full_estimator"].predict(
        np.asarray(
            [[float(row[name]) for name in lower_feature_names] for row in test_rows],
            dtype=np.float64,
        )
    )
    test_augmented = _attach_prediction_feature(
        test_rows,
        lower_prediction_feature_name,
        test_lower_predictions,
    )
    return {
        "dev_lower_rows": dev_augmented,
        "test_lower_rows": test_augmented,
        "test_lower_predictions": test_lower_predictions,
        "feature_importance_rows": list(lower_cache["feature_importance_rows"]),
    }


def _evaluate_upper_rf_candidate(
    model_spec: Dict[str, Any],
    candidate_name: str,
    candidate_params: Dict[str, Any],
    outer_fold_caches: Dict[int, Dict[str, Any]],
    cv_folds: Sequence[Dict[str, Any]],
    dev_indices: Sequence[int],
    global_categorical_levels: Dict[str, Sequence[str]],
    verbose: bool,
) -> Dict[str, Any]:
    continuous_feature_names = list(model_spec["continuous_feature_names"])
    categorical_feature_names = list(model_spec["categorical_feature_names"])
    uses_lower_model = bool(model_spec["uses_lower_model"])
    rf_feature_names = _build_rf_feature_names(
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )

    fold_rows: List[Dict[str, Any]] = []
    dev_prediction_lookup: Dict[int, float] = {}
    dev_row_lookup: Dict[int, Dict[str, Any]] = {}
    for outer_fold_position, fold in enumerate(cv_folds, start=1):
        cache = outer_fold_caches[int(fold["fold_index"])]
        train_rows = cache["train_lower_rows"] if uses_lower_model else cache["train_base_rows"]
        val_rows = cache["val_lower_rows"] if uses_lower_model else cache["val_base_rows"]

        if verbose:
            print(
                "[attempt3-study02] Candidate evaluation | {} | {} | max_depth={} | outer_fold={}/{} | train_rows={} | val_rows={}".format(
                    model_spec["model_name"],
                    candidate_name,
                    candidate_params.get("max_depth"),
                    outer_fold_position,
                    len(cv_folds),
                    len(train_rows),
                    len(val_rows),
                )
            )

        estimator = _fit_random_forest_estimator(candidate_params)
        x_train = _build_rf_matrix(
            train_rows,
            continuous_feature_names,
            categorical_feature_names,
            global_categorical_levels,
        )
        y_train = np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64)
        x_val = _build_rf_matrix(
            val_rows,
            continuous_feature_names,
            categorical_feature_names,
            global_categorical_levels,
        )
        estimator.fit(x_train, y_train)
        val_predictions = estimator.predict(x_val)
        fold_eval = evaluate_predictions(val_rows, val_predictions)
        if verbose:
            print(
                "[attempt3-study02] Completed candidate fold | {} | {} | outer_fold={}/{} | fold_mae={:.4f}".format(
                    model_spec["model_name"],
                    candidate_name,
                    outer_fold_position,
                    len(cv_folds),
                    float(fold_eval["overall"]["mae"]),
                )
            )
        fold_rows.append(
            {
                "fold_index": int(fold["fold_index"]),
                "overall": fold_eval["overall"],
                "per_distance_range": fold_eval["per_distance_range"],
            }
        )
        for global_index, row, prediction in zip(cache["global_val_indices"], val_rows, val_predictions):
            dev_prediction_lookup[int(global_index)] = float(prediction)
            dev_row_lookup[int(global_index)] = dict(row)

    ordered_dev_rows = [dev_row_lookup[int(index)] for index in dev_indices]
    ordered_predictions = np.asarray(
        [dev_prediction_lookup[int(index)] for index in dev_indices],
        dtype=np.float64,
    )
    oof_eval = evaluate_predictions(ordered_dev_rows, ordered_predictions)
    prediction_rows = _prediction_rows(
        rows=ordered_dev_rows,
        predictions=ordered_predictions,
        model_name="{}__{}".format(model_spec["model_name"], candidate_name),
        split_name="cv_oof",
        model_family="random_forest",
    )
    return {
        "model_name": str(model_spec["model_name"]),
        "candidate_name": str(candidate_name),
        "candidate_params": dict(candidate_params),
        "rf_feature_names": rf_feature_names,
        "fold_metrics": fold_rows,
        "summary": _cv_summary_from_fold_rows(fold_rows),
        "oof_overall": oof_eval["overall"],
        "oof_per_distance_range": oof_eval["per_distance_range"],
        "oof_prediction_rows": prediction_rows,
        "oof_rows": ordered_dev_rows,
        "oof_predictions": ordered_predictions,
    }


def _candidate_metric_row(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(result["summary"])
    oof_overall = dict(result["oof_overall"])
    row = {
        "model_name": result["model_name"],
        "candidate_name": result["candidate_name"],
        "num_features": len(result["rf_feature_names"]),
        "feature_names": " | ".join(result["rf_feature_names"]),
        "oof_mae": float(oof_overall["mae"]),
        "oof_mean_relative_error": float(oof_overall["mean_relative_error"]),
        "oof_rmse": float(oof_overall["rmse"]),
        "oof_r2": float(oof_overall["r2"]),
        "oof_mean_signed_error": float(oof_overall["mean_signed_error"]),
    }
    for key, value in summary.items():
        row[str(key)] = value
    for key, value in result["candidate_params"].items():
        row["param_{}".format(key)] = value
    return row


def _select_best_candidate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No RF candidate results were provided.")
    return sorted(
        results,
        key=lambda row: (
            float(row["oof_overall"]["mae"]),
            float(row["oof_overall"]["mean_relative_error"]),
            str(row["candidate_name"]),
        ),
    )[0]


def _fit_and_evaluate_selected_upper_rf(
    model_spec: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    global_categorical_levels: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    continuous_feature_names = list(model_spec["continuous_feature_names"])
    categorical_feature_names = list(model_spec["categorical_feature_names"])
    estimator = _fit_random_forest_estimator(selected_candidate["candidate_params"])
    x_train = _build_rf_matrix(
        train_rows,
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )
    y_train = np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64)
    x_test = _build_rf_matrix(
        test_rows,
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )
    estimator.fit(x_train, y_train)
    test_predictions = estimator.predict(x_test)
    test_eval = evaluate_predictions(test_rows, test_predictions)
    feature_names = _build_rf_feature_names(
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )
    importance_rows = [
        {
            "model_name": selected_candidate["model_name"],
            "candidate_name": selected_candidate["candidate_name"],
            "model_family": "random_forest",
            "term": feature_name,
            "importance_type": "feature_importance",
            "importance_value": float(importance_value),
        }
        for feature_name, importance_value in zip(feature_names, estimator.feature_importances_)
    ]
    prediction_rows = _prediction_rows(
        rows=test_rows,
        predictions=test_predictions,
        model_name="{}__{}".format(selected_candidate["model_name"], selected_candidate["candidate_name"]),
        split_name="test",
        model_family="random_forest",
    )
    return {
        "metrics": test_eval,
        "prediction_rows": prediction_rows,
        "importance_rows": importance_rows,
        "predictions": test_predictions,
        "feature_names": feature_names,
    }


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

    candidate_params_map = _candidate_params_by_name(resolved_config)
    model_specs = _build_model_specs(resolved_config, lower_prediction_feature_name)
    for spec in model_specs:
        if not spec["candidate_names"]:
            spec["candidate_names"] = list(candidate_params_map.keys())
        invalid_candidate_names = [
            name for name in spec["candidate_names"] if name not in candidate_params_map
        ]
        if invalid_candidate_names:
            raise ValueError(
                "Unknown upper RF candidates for model '{}': {}".format(
                    spec["model_name"],
                    invalid_candidate_names,
                )
            )

    feature_summary = dict(merge_summary)
    feature_summary.update(
        {
            "study_name": str(resolved_config.get("study_name", "study02")),
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
            "upper_rf_candidate_names": list(candidate_params_map.keys()),
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

    needs_lower_model = any(bool(spec["uses_lower_model"]) for spec in model_specs)
    lower_rf_params = dict(lower_model_config.get("random_forest", {}))
    lower_inner_num_folds = int(lower_model_config.get("inner_num_folds", 5))
    lower_inner_seed = int(lower_model_config.get("inner_seed", 0))

    if verbose:
        print(
            "[attempt3-study02] Stacked big-RF comparison\n"
            "  config: {}\n"
            "  merged_rows: {}\n"
            "  shared_rows_used: {}\n"
            "  attempt1_only_rows_dropped: {}\n"
            "  study06_only_rows_dropped: {}\n"
            "  selected_lower_model: {}\n"
            "  lower_feature_count: {}\n"
            "  upper_rf_candidates: {}\n"
            "  requested_models: {}".format(
                Path(config_path or DEFAULT_CONFIG_PATH),
                len(merged_rows),
                len(merged_rows),
                merge_summary["num_attempt1_rows_missing_from_study06"],
                merge_summary["num_extra_study06_rows"],
                selected_lower_model_name,
                len(selected_lower_feature_names),
                ", ".join(candidate_params_map.keys()),
                ", ".join(spec["model_name"] for spec in model_specs),
            )
        )

    global_categorical_levels = _build_global_categorical_levels(
        merged_rows,
        BASE_CATEGORICAL_FEATURE_NAMES,
    )

    outer_fold_caches = _prepare_outer_fold_caches(
        merged_rows=merged_rows,
        cv_folds=split_data["cv_folds"],
        lower_feature_names=selected_lower_feature_names,
        lower_prediction_feature_name=lower_prediction_feature_name,
        lower_rf_params=lower_rf_params,
        lower_inner_num_folds=lower_inner_num_folds,
        lower_inner_seed=lower_inner_seed,
        needs_lower_model=needs_lower_model,
        verbose=verbose,
    )

    cv_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study02")),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "models": {},
    }
    grouped_summary_rows: List[Dict[str, Any]] = []
    cv_prediction_rows: List[Dict[str, Any]] = []
    upper_candidate_rows: List[Dict[str, Any]] = []
    selected_model_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    selected_candidates: Dict[str, Dict[str, Any]] = {}

    for model_index, model_spec in enumerate(model_specs, start=1):
        if verbose:
            print(
                "[attempt3-study02] Candidate sweep {}/{} | {} | uses_lower_model={} | num_candidates={}".format(
                    model_index,
                    len(model_specs),
                    model_spec["model_name"],
                    bool(model_spec["uses_lower_model"]),
                    len(model_spec["candidate_names"]),
                )
            )
        candidate_results: List[Dict[str, Any]] = []
        for candidate_index, candidate_name in enumerate(model_spec["candidate_names"], start=1):
            candidate_params = candidate_params_map[candidate_name]
            if verbose:
                print(
                    "[attempt3-study02] Evaluating candidate {}/{} | {} | {}".format(
                        candidate_index,
                        len(model_spec["candidate_names"]),
                        model_spec["model_name"],
                        candidate_name,
                    )
                )
            result = _evaluate_upper_rf_candidate(
                model_spec=model_spec,
                candidate_name=candidate_name,
                candidate_params=candidate_params,
                outer_fold_caches=outer_fold_caches,
                cv_folds=split_data["cv_folds"],
                dev_indices=split_data["dev_indices"],
                global_categorical_levels=global_categorical_levels,
                verbose=verbose,
            )
            candidate_results.append(result)
            upper_candidate_rows.append(_candidate_metric_row(result))

        best_candidate = _select_best_candidate(candidate_results)
        selected_candidates[str(model_spec["model_name"])] = best_candidate
        selected_summary = dict(best_candidate["summary"])
        selected_model_rows.append(
            {
                "model_name": str(model_spec["model_name"]),
                "selected_candidate_name": str(best_candidate["candidate_name"]),
                "num_features": len(best_candidate["rf_feature_names"]),
                "feature_names": " | ".join(best_candidate["rf_feature_names"]),
                "oof_mae": float(best_candidate["oof_overall"]["mae"]),
                "oof_mean_relative_error": float(best_candidate["oof_overall"]["mean_relative_error"]),
                "oof_rmse": float(best_candidate["oof_overall"]["rmse"]),
                "oof_r2": float(best_candidate["oof_overall"]["r2"]),
                **selected_summary,
                **{
                    "param_{}".format(key): value
                    for key, value in best_candidate["candidate_params"].items()
                },
            }
        )
        cv_metrics["models"][str(model_spec["model_name"])] = {
            "selected_candidate_name": str(best_candidate["candidate_name"]),
            "selected_candidate_params": dict(best_candidate["candidate_params"]),
            "candidate_metric_rows": [
                _candidate_metric_row(result) for result in candidate_results
            ],
            "fold_metrics": best_candidate["fold_metrics"],
            "summary": dict(best_candidate["summary"]),
            "oof_overall": dict(best_candidate["oof_overall"]),
            "oof_per_distance_range": dict(best_candidate["oof_per_distance_range"]),
        }
        ranking_rows.append(
            {
                "model_name": str(model_spec["model_name"]),
                "selected_candidate_name": str(best_candidate["candidate_name"]),
                "oof_mae": float(best_candidate["oof_overall"]["mae"]),
                "oof_mean_relative_error": float(best_candidate["oof_overall"]["mean_relative_error"]),
            }
        )
        cv_prediction_rows.extend(best_candidate["oof_prediction_rows"])
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=best_candidate["oof_rows"],
                predictions=best_candidate["oof_predictions"],
                model_name="{}__{}".format(model_spec["model_name"], best_candidate["candidate_name"]),
                split_name="cv_oof",
            )
        )
        if verbose:
            print(
                "[attempt3-study02] Selected best candidate | {} | {} | oof_mae={:.4f}".format(
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                    float(best_candidate["oof_overall"]["mae"]),
                )
            )

    ranked_models = sorted(
        ranking_rows,
        key=lambda row: (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for rank_index, row in enumerate(ranked_models, start=1):
        row["rank"] = rank_index
    winner_model_name = str(ranked_models[0]["model_name"])
    winner_candidate_name = str(ranked_models[0]["selected_candidate_name"])
    if verbose:
        print(
            "[attempt3-study02] CV winner selected | {} | {} | oof_mae={:.4f}".format(
                winner_model_name,
                winner_candidate_name,
                float(ranked_models[0]["oof_mae"]),
            )
        )
    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["winner_candidate_name"] = winner_candidate_name
    cv_metrics["ranking"] = ranked_models
    _write_json(cv_metrics, reports_dir / "cv_metrics.json")
    _write_csv_rows(upper_candidate_rows, reports_dir / "upper_rf_candidate_metrics.csv")
    _write_csv_rows(selected_model_rows, reports_dir / "upper_rf_selected_models.csv")
    _write_csv_rows(cv_prediction_rows, reports_dir / "cv_predictions.csv")

    heldout_lower_cache: Optional[Dict[str, Any]] = None
    if needs_lower_model:
        heldout_lower_cache = _prepare_heldout_lower_cache(
            dev_rows=dev_rows,
            test_rows=test_rows,
            lower_feature_names=selected_lower_feature_names,
            lower_prediction_feature_name=lower_prediction_feature_name,
            lower_rf_params=lower_rf_params,
            lower_inner_num_folds=lower_inner_num_folds,
            lower_inner_seed=lower_inner_seed,
            verbose=verbose,
        )
    else:
        heldout_lower_cache = {
            "dev_lower_rows": dev_rows,
            "test_lower_rows": test_rows,
            "test_lower_predictions": None,
            "feature_importance_rows": [],
        }

    test_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study02")),
        "winner_model_name": winner_model_name,
        "winner_candidate_name": winner_candidate_name,
        "selected_lower_model_name": selected_lower_model_name,
        "models": {},
    }
    test_prediction_rows: List[Dict[str, Any]] = []
    upper_feature_importance_rows: List[Dict[str, Any]] = []

    for model_index, model_spec in enumerate(model_specs, start=1):
        best_candidate = selected_candidates[str(model_spec["model_name"])]
        if verbose:
            print(
                "[attempt3-study02] Held-out test fit {}/{} | {} | candidate={}".format(
                    model_index,
                    len(model_specs),
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                )
            )
        train_rows = (
            heldout_lower_cache["dev_lower_rows"] if model_spec["uses_lower_model"] else dev_rows
        )
        eval_rows = (
            heldout_lower_cache["test_lower_rows"] if model_spec["uses_lower_model"] else test_rows
        )
        test_result = _fit_and_evaluate_selected_upper_rf(
            model_spec=model_spec,
            selected_candidate=best_candidate,
            train_rows=train_rows,
            test_rows=eval_rows,
            global_categorical_levels=global_categorical_levels,
        )
        test_metrics["models"][str(model_spec["model_name"])] = {
            "selected_candidate_name": str(best_candidate["candidate_name"]),
            "selected_candidate_params": dict(best_candidate["candidate_params"]),
            "is_cv_winner": str(model_spec["model_name"]) == winner_model_name,
            "overall": dict(test_result["metrics"]["overall"]),
            "per_distance_range": dict(test_result["metrics"]["per_distance_range"]),
        }
        test_prediction_rows.extend(test_result["prediction_rows"])
        upper_feature_importance_rows.extend(test_result["importance_rows"])
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=eval_rows,
                predictions=test_result["predictions"],
                model_name="{}__{}".format(model_spec["model_name"], best_candidate["candidate_name"]),
                split_name="test",
            )
        )
        if verbose:
            print(
                "[attempt3-study02] Completed held-out test fit | {} | candidate={} | test_mae={:.4f}".format(
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                    float(test_result["metrics"]["overall"]["mae"]),
                )
            )

    _write_json(test_metrics, reports_dir / "test_metrics.json")
    _write_csv_rows(test_prediction_rows, reports_dir / "test_predictions.csv")
    _write_csv_rows(grouped_summary_rows, reports_dir / "grouped_error_summary.csv")
    _write_csv_rows(upper_feature_importance_rows, reports_dir / "upper_rf_feature_importances.csv")
    _write_csv_rows(
        list(heldout_lower_cache.get("feature_importance_rows", [])),
        reports_dir / "lower_model_feature_importances.csv",
    )
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

    winner_test_metrics = test_metrics["models"][winner_model_name]["overall"]
    selected_candidate_map = {
        model_name: selected_candidates[model_name]["candidate_name"] for model_name in selected_candidates
    }
    did_unbounded_win = any(
        str(candidate_name) == "rf_unbounded" for candidate_name in selected_candidate_map.values()
    )
    report_summary = {
        "study_name": str(resolved_config.get("study_name", "study02")),
        "num_merged_rows": len(merged_rows),
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "num_selected_lower_features": len(selected_lower_feature_names),
        "winner_model_name": winner_model_name,
        "winner_candidate_name": winner_candidate_name,
        "winner_cv_oof_mae": float(ranked_models[0]["oof_mae"]),
        "winner_cv_oof_mean_relative_error": float(ranked_models[0]["oof_mean_relative_error"]),
        "winner_test_mae": float(winner_test_metrics["mae"]),
        "winner_test_mean_relative_error": float(winner_test_metrics["mean_relative_error"]),
        "selected_upper_candidates_by_model": selected_candidate_map,
        "did_unbounded_depth_win_any_model": bool(did_unbounded_win),
        "merged_features_csv": str(merged_features_csv),
        "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
        "test_metrics_json": str(reports_dir / "test_metrics.json"),
        "upper_rf_candidate_metrics_csv": str(reports_dir / "upper_rf_candidate_metrics.csv"),
    }
    _write_json(report_summary, reports_dir / "summary.json")

    if verbose:
        print(
            "[attempt3-study02] Study complete\n"
            "  winner_model: {}\n"
            "  winner_candidate: {}\n"
            "  winner_test_mae: {:.4f}\n"
            "  summary_json: {}".format(
                winner_model_name,
                winner_candidate_name,
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
