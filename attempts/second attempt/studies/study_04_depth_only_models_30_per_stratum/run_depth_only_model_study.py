"""
Run the Study 04 depth-only lower-model search on the cached multiscale features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.depth_representation_features import (
    DEPTH_SCORE_FIELD_NAMES,
    read_depth_representation_records_csv,
)
from pipeline.calibration.depth_representation_study import (
    _pearson_correlation,
    _spearman_correlation,
    build_method_id,
)
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_csv_rows(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _standardize_train_matrix(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    return (x_train - means) / scales, means, scales


def _fit_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    ridge_alpha: float = 0.0,
) -> Dict[str, Any]:
    x_train_standardized, means, scales = _standardize_train_matrix(x_train)
    design = np.column_stack([np.ones(x_train_standardized.shape[0], dtype=np.float64), x_train_standardized])
    if model_type == "ols":
        coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    elif model_type == "ridge":
        alpha = float(ridge_alpha)
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y_train)
    else:
        raise ValueError("Unsupported linear model_type '{}'.".format(model_type))
    intercept_std = float(coefficients[0])
    coef_std = np.asarray(coefficients[1:], dtype=np.float64)
    coef_original = coef_std / scales
    intercept_original = intercept_std - float(np.sum(coef_std * means / scales))
    return {
        "intercept": float(intercept_original),
        "coefficients": coef_original.astype(np.float64),
        "means": means.astype(np.float64),
        "scales": scales.astype(np.float64),
    }


def _predict_linear_regression(x_values: np.ndarray, fit_result: Dict[str, Any]) -> np.ndarray:
    return float(fit_result["intercept"]) + np.asarray(x_values, dtype=np.float64) @ np.asarray(
        fit_result["coefficients"], dtype=np.float64
    )


def _load_optional_dependencies() -> Dict[str, Any]:
    deps: Dict[str, Any] = {"sklearn": False, "xgboost": False}
    try:
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

        deps.update(
            {
                "sklearn": True,
                "RandomForestRegressor": RandomForestRegressor,
                "ExtraTreesRegressor": ExtraTreesRegressor,
                "GradientBoostingRegressor": GradientBoostingRegressor,
            }
        )
    except Exception as exc:  # pragma: no cover - dependency availability is environment-specific
        deps["sklearn_error"] = str(exc)

    try:
        from xgboost import XGBRegressor

        deps.update({"xgboost": True, "XGBRegressor": XGBRegressor})
    except Exception as exc:  # pragma: no cover - dependency availability is environment-specific
        deps["xgboost_error"] = str(exc)

    return deps


def _pivot_records_to_image_rows(records: Sequence[Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows_by_image: Dict[str, Dict[str, Any]] = {}
    feature_names: List[str] = []
    for record in records:
        image_key = str(record.image_path)
        row = rows_by_image.setdefault(
            image_key,
            {
                "image_path": str(record.image_path),
                "label_path": str(record.label_path),
                "stratum_key": str(record.stratum_key),
                "selection_rank_within_stratum": int(record.selection_rank_within_stratum),
                "true_distance_m": float(record.true_distance_m),
                "weather": str(record.weather),
                "time_of_day": str(record.time_of_day),
            },
        )
        for score_field in DEPTH_SCORE_FIELD_NAMES:
            method_id = build_method_id(record.context_window, record.aggregation_method, score_field)
            row[method_id] = float(getattr(record, score_field))
            if method_id not in feature_names:
                feature_names.append(method_id)

    rows = sorted(
        rows_by_image.values(),
        key=lambda row: (
            float(row["true_distance_m"]),
            str(row["weather"]),
            str(row["time_of_day"]),
            str(row["image_path"]),
        ),
    )
    for row in rows:
        for feature_name in feature_names:
            if feature_name not in row:
                raise ValueError("Missing feature '{}' in fused row.".format(feature_name))
    return rows, feature_names


def _make_balanced_folds(
    rows: Sequence[Dict[str, Any]],
    num_folds: int,
    seed: int,
) -> List[np.ndarray]:
    grouped_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(str(row["stratum_key"]), []).append(index)
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for stratum_key in sorted(grouped_indices):
        indices = sorted(
            grouped_indices[stratum_key],
            key=lambda idx: (
                int(rows[idx]["selection_rank_within_stratum"]),
                str(rows[idx]["image_path"]),
            ),
        )
        permuted = list(indices)
        rng.shuffle(permuted)
        for rank, row_index in enumerate(permuted):
            folds[rank % num_folds].append(row_index)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def _bootstrap_metric_interval(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    metric_name: str,
    num_iterations: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = np.arange(true_values.size)
    samples = []
    for _ in range(num_iterations):
        sample_indices = rng.choice(indices, size=indices.size, replace=True)
        sample_true = true_values[sample_indices]
        sample_pred = predicted_values[sample_indices]
        metrics = evaluate_predictions(
            [{"true_distance_m": float(value), "weather": "", "time_of_day": ""} for value in sample_true],
            sample_pred,
        )["overall"]
        samples.append(float(metrics[metric_name]))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _bootstrap_paired_mae_delta(
    true_values: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    num_iterations: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = np.arange(true_values.size)
    deltas = []
    for _ in range(num_iterations):
        sample_indices = rng.choice(indices, size=indices.size, replace=True)
        err_a = np.abs(predictions_a[sample_indices] - true_values[sample_indices])
        err_b = np.abs(predictions_b[sample_indices] - true_values[sample_indices])
        deltas.append(float(np.mean(err_a) - np.mean(err_b)))
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _fit_tree_estimator(
    model_family: str,
    params: Dict[str, Any],
    dependencies: Dict[str, Any],
):
    if model_family == "random_forest":
        estimator_cls = dependencies["RandomForestRegressor"]
        return estimator_cls(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=params.get("max_features", "sqrt"),
            random_state=int(params.get("random_state", 0)),
            n_jobs=-1,
        )
    if model_family == "extra_trees":
        estimator_cls = dependencies["ExtraTreesRegressor"]
        return estimator_cls(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=params.get("max_features", "sqrt"),
            random_state=int(params.get("random_state", 0)),
            n_jobs=-1,
        )
    if model_family == "gradient_boosting":
        estimator_cls = dependencies["GradientBoostingRegressor"]
        return estimator_cls(
            n_estimators=int(params.get("n_estimators", 250)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=int(params.get("random_state", 0)),
        )
    if model_family == "xgboost":
        estimator_cls = dependencies["XGBRegressor"]
        return estimator_cls(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 4)),
            min_child_weight=float(params.get("min_child_weight", 2.0)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            random_state=int(params.get("random_state", 0)),
            n_jobs=-1,
            objective="reg:squarederror",
        )
    raise ValueError("Unsupported tree model family '{}'.".format(model_family))


def _evaluate_single_feature(
    rows: Sequence[Dict[str, Any]],
    feature_name: str,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    x_values = np.asarray([float(row[feature_name]) for row in rows], dtype=np.float64)
    predictions = np.zeros(true_values.shape[0], dtype=np.float64)
    for fold_indices in folds:
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[fold_indices] = False
        fit_result = _fit_linear_regression(
            x_train=x_values[train_mask].reshape(-1, 1),
            y_train=true_values[train_mask],
            model_type="ols",
            ridge_alpha=0.0,
        )
        predictions[fold_indices] = _predict_linear_regression(x_values[fold_indices].reshape(-1, 1), fit_result)
    evaluation = evaluate_predictions(rows, predictions)
    return {
        "method_id": feature_name,
        "count": evaluation["overall"]["count"],
        "pearson_correlation": _pearson_correlation(x_values, true_values),
        "spearman_correlation": _spearman_correlation(x_values, true_values),
        "cv_mae": evaluation["overall"]["mae"],
        "cv_median_absolute_error": evaluation["overall"]["median_absolute_error"],
        "cv_mean_relative_error": evaluation["overall"]["mean_relative_error"],
        "cv_median_relative_error": evaluation["overall"]["median_relative_error"],
        "cv_rmse": evaluation["overall"]["rmse"],
        "cv_r2": evaluation["overall"]["r2"],
        "cv_mean_signed_error": evaluation["overall"]["mean_signed_error"],
        "cv_median_signed_error": evaluation["overall"]["median_signed_error"],
        "cv_p90_absolute_error": evaluation["overall"]["p90_absolute_error"],
        "cv_p95_absolute_error": evaluation["overall"]["p95_absolute_error"],
        "cv_max_absolute_error": evaluation["overall"]["max_absolute_error"],
        "cv_within_5m_rate": evaluation["overall"]["within_5m_rate"],
        "cv_within_10m_rate": evaluation["overall"]["within_10m_rate"],
        "cv_within_20m_rate": evaluation["overall"]["within_20m_rate"],
        "cv_overprediction_rate": evaluation["overall"]["overprediction_rate"],
        "cv_underprediction_rate": evaluation["overall"]["underprediction_rate"],
        "predictions": predictions,
    }


def _evaluate_linear_model(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    folds: Sequence[np.ndarray],
    model_name: str,
    model_type: str,
    ridge_alpha: float,
    verbose: bool,
) -> Dict[str, Any]:
    x_values = np.asarray(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=np.float64,
    )
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    predictions = np.zeros(true_values.shape[0], dtype=np.float64)
    for fold_index, fold_indices in enumerate(folds, start=1):
        if verbose:
            print(
                "[study-04]     fold {}/{} | {} | {} feature(s)".format(
                    fold_index,
                    len(folds),
                    model_name,
                    len(feature_names),
                )
            )
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[fold_indices] = False
        fit_result = _fit_linear_regression(
            x_train=x_values[train_mask],
            y_train=true_values[train_mask],
            model_type=model_type,
            ridge_alpha=ridge_alpha,
        )
        predictions[fold_indices] = _predict_linear_regression(x_values[fold_indices], fit_result)

    full_fit = _fit_linear_regression(x_values, true_values, model_type=model_type, ridge_alpha=ridge_alpha)
    evaluation = evaluate_predictions(rows, predictions)
    coefficient_rows = [
        {
            "model_name": model_name,
            "model_family": model_type,
            "term": "intercept",
            "importance_type": "coefficient",
            "importance_value": float(full_fit["intercept"]),
        }
    ]
    for feature_name, coefficient in zip(feature_names, full_fit["coefficients"]):
        coefficient_rows.append(
            {
                "model_name": model_name,
                "model_family": model_type,
                "term": feature_name,
                "importance_type": "coefficient",
                "importance_value": float(coefficient),
            }
        )
    return {
        "model_name": model_name,
        "model_family": model_type,
        "feature_names": list(feature_names),
        "predictions": predictions,
        "metrics": evaluation["overall"],
        "grouped_rows": build_grouped_error_summary(rows, predictions, model_name=model_name, split_name="study_04_cv"),
        "importance_rows": coefficient_rows,
    }


def _evaluate_tree_model(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    folds: Sequence[np.ndarray],
    model_name: str,
    model_family: str,
    params: Dict[str, Any],
    dependencies: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    x_values = np.asarray(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=np.float64,
    )
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    predictions = np.zeros(true_values.shape[0], dtype=np.float64)
    for fold_index, fold_indices in enumerate(folds, start=1):
        if verbose:
            print(
                "[study-04]     fold {}/{} | {} | {} feature(s)".format(
                    fold_index,
                    len(folds),
                    model_name,
                    len(feature_names),
                )
            )
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[fold_indices] = False
        estimator = _fit_tree_estimator(model_family, params, dependencies)
        estimator.fit(x_values[train_mask], true_values[train_mask])
        predictions[fold_indices] = estimator.predict(x_values[fold_indices])

    estimator = _fit_tree_estimator(model_family, params, dependencies)
    estimator.fit(x_values, true_values)
    evaluation = evaluate_predictions(rows, predictions)
    importance_rows = []
    if hasattr(estimator, "feature_importances_"):
        for feature_name, importance_value in zip(feature_names, estimator.feature_importances_):
            importance_rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "term": feature_name,
                    "importance_type": "feature_importance",
                    "importance_value": float(importance_value),
                }
            )
    return {
        "model_name": model_name,
        "model_family": model_family,
        "feature_names": list(feature_names),
        "predictions": predictions,
        "metrics": evaluation["overall"],
        "grouped_rows": build_grouped_error_summary(rows, predictions, model_name=model_name, split_name="study_04_cv"),
        "importance_rows": importance_rows,
    }


def _variance_filter(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> List[str]:
    kept = []
    for feature_name in feature_names:
        values = np.asarray([float(row[feature_name]) for row in rows], dtype=np.float64)
        if np.std(values) > 1e-12:
            kept.append(feature_name)
    return kept


def _generate_candidate_models(
    rows: Sequence[Dict[str, Any]],
    available_feature_names: Sequence[str],
    single_feature_rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    dependencies: Dict[str, Any],
    folds: Sequence[np.ndarray],
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    modeling_config = config.get("depth_only_modeling", {})
    study_config = config.get("study", {})
    context_names = [str(item["name"]) for item in study_config.get("context_windows", [])]
    top_pool_size = int(modeling_config.get("top_pool_size", 24))
    greedy_max_features = int(modeling_config.get("greedy_forward_max_features", 8))
    top_pool = [row["method_id"] for row in single_feature_rows[:top_pool_size]]

    candidate_specs: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"greedy_paths": {}, "skipped_models": []}

    def add_linear(model_name: str, feature_names: Sequence[str], model_type: str, ridge_alpha: float = 0.0) -> None:
        filtered = _variance_filter(rows, [name for name in feature_names if name in set(available_feature_names)])
        if filtered:
            candidate_specs.append(
                {
                    "model_name": model_name,
                    "model_kind": "linear",
                    "model_family": model_type,
                    "feature_names": filtered,
                    "ridge_alpha": float(ridge_alpha),
                }
            )

    def add_tree(model_name: str, feature_names: Sequence[str], model_family: str, params: Dict[str, Any]) -> None:
        filtered = _variance_filter(rows, [name for name in feature_names if name in set(available_feature_names)])
        if filtered:
            candidate_specs.append(
                {
                    "model_name": model_name,
                    "model_kind": "tree",
                    "model_family": model_family,
                    "feature_names": filtered,
                    "params": dict(params),
                }
            )

    for top_k in (3, 5, 8, 12):
        features = [row["method_id"] for row in single_feature_rows[:top_k]]
        add_linear("top_{}_ols".format(top_k), features, model_type="ols")
        add_linear("top_{}_ridge_alpha_1".format(top_k), features, model_type="ridge", ridge_alpha=1.0)
        add_linear("top_{}_ridge_alpha_10".format(top_k), features, model_type="ridge", ridge_alpha=10.0)

    best_by_context = []
    for context_name in context_names:
        matching_rows = [row for row in single_feature_rows if row["context_window"] == context_name]
        if matching_rows:
            best_by_context.append(matching_rows[0]["method_id"])
    add_linear("best_per_context_ols", best_by_context, model_type="ols")
    add_linear("best_per_context_ridge_alpha_1", best_by_context, model_type="ridge", ridge_alpha=1.0)

    best_by_score_field = []
    for score_field in DEPTH_SCORE_FIELD_NAMES:
        matching_rows = [row for row in single_feature_rows if row["score_field"] == score_field]
        if matching_rows:
            best_by_score_field.append(matching_rows[0]["method_id"])
    add_linear("best_per_score_field_ols", best_by_score_field, model_type="ols")

    local_shortlist = [
        build_method_id("bbox_only", "bbox_midpoint", "object_depth"),
        build_method_id("bbox_only", "bbox_midpoint", "object_depth_percentile_5_95"),
        build_method_id("bbox_expand_1_5x", "bbox_midpoint", "object_minus_ring"),
        build_method_id("bbox_expand_2x", "bbox_midpoint", "object_minus_ring"),
        build_method_id("bbox_expand_2x", "inner50_median", "object_minus_ring"),
    ]
    add_linear("local_shortlist_ols", local_shortlist, model_type="ols")
    add_linear("local_shortlist_ridge_alpha_1", local_shortlist, model_type="ridge", ridge_alpha=1.0)

    for model_type, ridge_alpha in (("ols", 0.0), ("ridge", 1.0)):
        selected: List[str] = []
        path_rows: List[Dict[str, Any]] = []
        best_mae: Optional[float] = None
        for step in range(1, greedy_max_features + 1):
            best_candidate_feature = None
            best_candidate_result = None
            for candidate_feature in top_pool:
                if candidate_feature in selected:
                    continue
                result = _evaluate_linear_model(
                    rows=rows,
                    feature_names=selected + [candidate_feature],
                    folds=folds,
                    model_name="greedy_search_temp",
                    model_type=model_type,
                    ridge_alpha=ridge_alpha,
                    verbose=False,
                )
                candidate_mae = float(result["metrics"]["mae"])
                if best_candidate_result is None or candidate_mae < float(best_candidate_result["metrics"]["mae"]):
                    best_candidate_result = result
                    best_candidate_feature = candidate_feature
            if best_candidate_feature is None or best_candidate_result is None:
                break
            candidate_mae = float(best_candidate_result["metrics"]["mae"])
            if best_mae is not None and candidate_mae >= best_mae - 1e-6:
                break
            selected.append(best_candidate_feature)
            best_mae = candidate_mae
            model_name = "greedy_forward_{}_k{}".format(model_type, len(selected))
            add_linear(model_name, selected, model_type=model_type, ridge_alpha=ridge_alpha)
            path_rows.append(
                {
                    "step": len(selected),
                    "selected_feature": best_candidate_feature,
                    "cv_mae": candidate_mae,
                    "feature_names": list(selected),
                }
            )
        diagnostics["greedy_paths"]["{}_alpha_{}".format(model_type, ridge_alpha)] = path_rows

    tree_config = modeling_config.get("tree_models", {})
    if tree_config.get("enabled", True):
        top_feature_pool_sizes = [int(value) for value in tree_config.get("top_feature_pool_sizes", [12, 24])]
        if dependencies.get("sklearn", False):
            for top_k in top_feature_pool_sizes:
                features = [row["method_id"] for row in single_feature_rows[:top_k]]
                rf_params = dict(tree_config.get("random_forest", {}))
                rf_params["random_state"] = int(tree_config.get("random_state", 0))
                add_tree("random_forest_top_{}".format(top_k), features, "random_forest", rf_params)
                et_params = dict(tree_config.get("extra_trees", {}))
                et_params["random_state"] = int(tree_config.get("random_state", 0))
                add_tree("extra_trees_top_{}".format(top_k), features, "extra_trees", et_params)
                gb_params = dict(tree_config.get("gradient_boosting", {}))
                gb_params["random_state"] = int(tree_config.get("random_state", 0))
                add_tree("gradient_boosting_top_{}".format(top_k), features, "gradient_boosting", gb_params)
        else:
            diagnostics["skipped_models"].append(
                {
                    "model_group": "sklearn_tree_models",
                    "reason": dependencies.get("sklearn_error", "scikit-learn is not installed."),
                }
            )
        if dependencies.get("xgboost", False):
            for top_k in top_feature_pool_sizes:
                features = [row["method_id"] for row in single_feature_rows[:top_k]]
                xgb_params = dict(tree_config.get("xgboost", {}))
                xgb_params["random_state"] = int(tree_config.get("random_state", 0))
                add_tree("xgboost_top_{}".format(top_k), features, "xgboost", xgb_params)
        else:
            diagnostics["skipped_models"].append(
                {
                    "model_group": "xgboost_models",
                    "reason": dependencies.get("xgboost_error", "xgboost is not installed."),
                }
            )

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for spec in candidate_specs:
        dedupe_key = (
            spec["model_kind"],
            spec["model_family"],
            tuple(spec["feature_names"]),
            json.dumps(spec.get("params", {}), sort_keys=True),
            float(spec.get("ridge_alpha", 0.0)),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(spec)

    return deduped, diagnostics


def run_depth_only_model_study(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_csv_path = output_root / "features" / "representation_records.csv"
    if not features_csv_path.exists():
        raise FileNotFoundError(
            "Expected feature cache at {}. Run run_extend_feature_cache.py first.".format(features_csv_path)
        )

    fused_features_dir = output_root / "fused_features"
    reports_dir = output_root / "reports"
    fused_features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dependencies = _load_optional_dependencies()
    records = read_depth_representation_records_csv(features_csv_path)
    image_rows, available_feature_names = _pivot_records_to_image_rows(records)
    fused_features_path = fused_features_dir / "depth_only_feature_table.csv"
    _write_csv_rows(image_rows, fused_features_path)

    modeling_config = resolved_config.get("depth_only_modeling", {})
    num_folds = int(modeling_config.get("num_folds", 5))
    cv_seed = int(modeling_config.get("cv_seed", 0))
    folds = _make_balanced_folds(image_rows, num_folds=num_folds, seed=cv_seed)

    if verbose:
        print(
            "[study-04] Starting depth-only model study\n"
            "  config: {}\n"
            "  features_csv: {}\n"
            "  fused_features_csv: {}\n"
            "  num_image_rows: {}\n"
            "  num_available_depth_features: {}\n"
            "  num_folds: {}\n"
            "  sklearn_available: {}\n"
            "  xgboost_available: {}".format(
                config_path or DEFAULT_CONFIG_PATH,
                features_csv_path,
                fused_features_path,
                len(image_rows),
                len(available_feature_names),
                num_folds,
                dependencies.get("sklearn", False),
                dependencies.get("xgboost", False),
            )
        )

    single_feature_rows: List[Dict[str, Any]] = []
    single_feature_prediction_map: Dict[str, np.ndarray] = {}
    progress_every = max(1, int(resolved_config.get("study", {}).get("progress_log_every_single_features", 10)))
    for feature_index, feature_name in enumerate(available_feature_names, start=1):
        if verbose and (feature_index == 1 or feature_index == len(available_feature_names) or feature_index % progress_every == 0):
            print(
                "[study-04] Single-feature baseline {}/{}\n"
                "  feature: {}".format(
                    feature_index,
                    len(available_feature_names),
                    feature_name,
                )
            )
        result = _evaluate_single_feature(image_rows, feature_name, folds)
        predictions = result.pop("predictions")
        single_feature_rows.append(result)
        single_feature_prediction_map[feature_name] = predictions

    single_feature_rows = sorted(
        single_feature_rows,
        key=lambda row: (
            float(row["cv_mae"]),
            float(row["cv_mean_relative_error"]),
            -float(row["spearman_correlation"]),
            str(row["method_id"]),
        ),
    )
    for rank_index, row in enumerate(single_feature_rows, start=1):
        row["rank"] = rank_index
    for row in single_feature_rows:
        context_window, aggregation_method, score_field = str(row["method_id"]).split("__", 2)
        row["context_window"] = context_window
        row["aggregation_method"] = aggregation_method
        row["score_field"] = score_field

    best_single_feature = dict(single_feature_rows[0])

    _write_csv_rows(single_feature_rows, reports_dir / "single_feature_cv_metrics.csv")

    candidate_specs, diagnostics = _generate_candidate_models(
        rows=image_rows,
        available_feature_names=available_feature_names,
        single_feature_rows=single_feature_rows,
        config=resolved_config,
        dependencies=dependencies,
        folds=folds,
        verbose=verbose,
    )

    if verbose:
        print(
            "[study-04] Candidate models prepared\n"
            "  num_candidates: {}\n"
            "  best_single_feature: {}\n"
            "  best_single_feature_cv_mae: {:.4f}".format(
                len(candidate_specs),
                best_single_feature["method_id"],
                float(best_single_feature["cv_mae"]),
            )
        )

    model_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    grouped_rows: List[Dict[str, Any]] = []
    model_prediction_map: Dict[str, np.ndarray] = {}

    true_values = np.asarray([float(row["true_distance_m"]) for row in image_rows], dtype=np.float64)

    for candidate_index, spec in enumerate(candidate_specs, start=1):
        if verbose:
            print(
                "[study-04] Evaluating model {}/{}\n"
                "  model_name: {}\n"
                "  model_kind: {}\n"
                "  model_family: {}\n"
                "  feature_count: {}".format(
                    candidate_index,
                    len(candidate_specs),
                    spec["model_name"],
                    spec["model_kind"],
                    spec["model_family"],
                    len(spec["feature_names"]),
                )
            )
        if spec["model_kind"] == "linear":
            result = _evaluate_linear_model(
                rows=image_rows,
                feature_names=spec["feature_names"],
                folds=folds,
                model_name=spec["model_name"],
                model_type=spec["model_family"],
                ridge_alpha=float(spec.get("ridge_alpha", 0.0)),
                verbose=verbose,
            )
        else:
            result = _evaluate_tree_model(
                rows=image_rows,
                feature_names=spec["feature_names"],
                folds=folds,
                model_name=spec["model_name"],
                model_family=spec["model_family"],
                params=spec.get("params", {}),
                dependencies=dependencies,
                verbose=verbose,
            )
        predictions = np.asarray(result["predictions"], dtype=np.float64)
        model_prediction_map[spec["model_name"]] = predictions
        metrics = result["metrics"]
        model_rows.append(
            {
                "model_name": spec["model_name"],
                "model_kind": spec["model_kind"],
                "model_family": spec["model_family"],
                "feature_set_size": len(spec["feature_names"]),
                "feature_names": " | ".join(spec["feature_names"]),
                "cv_mae": metrics["mae"],
                "cv_median_absolute_error": metrics["median_absolute_error"],
                "cv_mean_relative_error": metrics["mean_relative_error"],
                "cv_median_relative_error": metrics["median_relative_error"],
                "cv_rmse": metrics["rmse"],
                "cv_r2": metrics["r2"],
                "cv_mean_signed_error": metrics["mean_signed_error"],
                "cv_median_signed_error": metrics["median_signed_error"],
                "cv_p90_absolute_error": metrics["p90_absolute_error"],
                "cv_p95_absolute_error": metrics["p95_absolute_error"],
                "cv_max_absolute_error": metrics["max_absolute_error"],
                "cv_within_5m_rate": metrics["within_5m_rate"],
                "cv_within_10m_rate": metrics["within_10m_rate"],
                "cv_within_20m_rate": metrics["within_20m_rate"],
                "cv_overprediction_rate": metrics["overprediction_rate"],
                "cv_underprediction_rate": metrics["underprediction_rate"],
            }
        )
        importance_rows.extend(result["importance_rows"])
        grouped_rows.extend(result["grouped_rows"])
        for row, prediction in zip(image_rows, predictions):
            signed_error = float(prediction) - float(row["true_distance_m"])
            prediction_rows.append(
                {
                    "model_name": spec["model_name"],
                    "model_kind": spec["model_kind"],
                    "model_family": spec["model_family"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                    "stratum_key": row["stratum_key"],
                    "selection_rank_within_stratum": int(row["selection_rank_within_stratum"]),
                    "weather": row["weather"],
                    "time_of_day": row["time_of_day"],
                    "true_distance_m": float(row["true_distance_m"]),
                    "predicted_distance_m": float(prediction),
                    "signed_error_m": signed_error,
                    "absolute_error_m": abs(signed_error),
                    "relative_error": abs(signed_error) / float(row["true_distance_m"]),
                }
            )

    model_rows = sorted(
        model_rows,
        key=lambda row: (
            float(row["cv_mae"]),
            float(row["cv_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for rank_index, row in enumerate(model_rows, start=1):
        row["rank"] = rank_index

    best_model = dict(model_rows[0]) if model_rows else {}
    best_linear_model = next((dict(row) for row in model_rows if row["model_kind"] == "linear" and row["feature_set_size"] > 1), {})
    best_tree_model = next((dict(row) for row in model_rows if row["model_kind"] == "tree"), {})

    bootstrap_iterations = int(modeling_config.get("bootstrap_iterations", 1000))
    bootstrap_seed = int(modeling_config.get("bootstrap_seed", 0))
    ci_rows: List[Dict[str, Any]] = []
    top_model_names = [best_single_feature["method_id"]]
    if best_model:
        top_model_names.append(best_model["model_name"])
    if best_linear_model:
        top_model_names.append(best_linear_model["model_name"])
    if best_tree_model:
        top_model_names.append(best_tree_model["model_name"])
    for model_name in dict.fromkeys(top_model_names):
        if model_name == best_single_feature["method_id"]:
            predictions = single_feature_prediction_map[model_name]
            model_kind = "single_feature"
        else:
            predictions = model_prediction_map[model_name]
            model_kind = next(row["model_kind"] for row in model_rows if row["model_name"] == model_name)
        for metric_name in ("mae", "rmse", "mean_relative_error"):
            ci_low, ci_high = _bootstrap_metric_interval(
                true_values=true_values,
                predicted_values=np.asarray(predictions, dtype=np.float64),
                metric_name=metric_name,
                num_iterations=bootstrap_iterations,
                seed=bootstrap_seed + len(ci_rows) + 1,
            )
            point_metrics = evaluate_predictions(image_rows, predictions)["overall"]
            ci_rows.append(
                {
                    "model_name": model_name,
                    "model_kind": model_kind,
                    "metric_name": metric_name,
                    "point_estimate": float(point_metrics[metric_name]),
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )

    comparison_rows: List[Dict[str, Any]] = []
    comparisons = []
    if best_model:
        comparisons.append((best_model["model_name"], best_single_feature["method_id"], "best_model_vs_best_single"))
    if best_tree_model and best_linear_model:
        comparisons.append((best_tree_model["model_name"], best_linear_model["model_name"], "best_tree_vs_best_linear"))
    for model_a, model_b, comparison_name in comparisons:
        pred_a = model_prediction_map.get(model_a, single_feature_prediction_map.get(model_a))
        pred_b = model_prediction_map.get(model_b, single_feature_prediction_map.get(model_b))
        mean_delta, ci_low, ci_high = _bootstrap_paired_mae_delta(
            true_values=true_values,
            predictions_a=np.asarray(pred_a, dtype=np.float64),
            predictions_b=np.asarray(pred_b, dtype=np.float64),
            num_iterations=bootstrap_iterations,
            seed=bootstrap_seed + 100 + len(comparison_rows),
        )
        err_a = np.abs(np.asarray(pred_a, dtype=np.float64) - true_values)
        err_b = np.abs(np.asarray(pred_b, dtype=np.float64) - true_values)
        comparison_rows.append(
            {
                "comparison_name": comparison_name,
                "model_a": model_a,
                "model_b": model_b,
                "mean_mae_delta_a_minus_b": mean_delta,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "a_better_rate": float(np.mean(err_a < err_b)),
                "b_better_rate": float(np.mean(err_b < err_a)),
                "tie_rate": float(np.mean(np.isclose(err_a, err_b))),
                "bootstrap_iterations": bootstrap_iterations,
            }
        )

    summary = {
        "study_name": resolved_config.get("study_name", output_root.name),
        "goal": modeling_config.get("goal", ""),
        "num_image_rows": len(image_rows),
        "num_available_depth_features": len(available_feature_names),
        "num_candidate_models": len(model_rows),
        "num_folds": num_folds,
        "dependencies": {
            "sklearn_available": dependencies.get("sklearn", False),
            "xgboost_available": dependencies.get("xgboost", False),
            "sklearn_error": dependencies.get("sklearn_error", ""),
            "xgboost_error": dependencies.get("xgboost_error", ""),
        },
        "best_single_feature": best_single_feature,
        "best_model": best_model,
        "best_linear_multifeature_model": best_linear_model,
        "best_tree_model": best_tree_model,
        "skipped_model_groups": diagnostics.get("skipped_models", []),
        "gap_best_model_vs_best_single": {
            "cv_mae": float(best_model.get("cv_mae", 0.0)) - float(best_single_feature["cv_mae"]) if best_model else 0.0,
            "cv_rmse": float(best_model.get("cv_rmse", 0.0)) - float(best_single_feature["cv_rmse"]) if best_model else 0.0,
            "cv_mean_relative_error": float(best_model.get("cv_mean_relative_error", 0.0))
            - float(best_single_feature["cv_mean_relative_error"])
            if best_model
            else 0.0,
        },
        "top_10_single_features": single_feature_rows[:10],
        "top_10_models": model_rows[:10],
        "greedy_search_diagnostics": diagnostics.get("greedy_paths", {}),
    }

    summary_path = reports_dir / "depth_only_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv_rows(model_rows, reports_dir / "depth_only_model_metrics.csv")
    _write_csv_rows(importance_rows, reports_dir / "depth_only_model_importances.csv")
    _write_csv_rows(prediction_rows, reports_dir / "depth_only_model_predictions.csv")
    _write_csv_rows(grouped_rows, reports_dir / "depth_only_grouped_error_summary.csv")
    _write_csv_rows(ci_rows, reports_dir / "depth_only_bootstrap_intervals.csv")
    _write_csv_rows(comparison_rows, reports_dir / "depth_only_paired_model_comparisons.csv")
    _write_csv_rows(
        [row for row in grouped_rows if row.get("grouping") == "distance_range"],
        reports_dir / "depth_only_distance_range_metrics.csv",
    )

    if verbose:
        print(
            "[study-04] Depth-only model study complete\n"
            "  best_single_feature: {}\n"
            "  best_single_feature_cv_mae: {:.4f}\n"
            "  best_model: {}\n"
            "  best_model_cv_mae: {:.4f}\n"
            "  gap_vs_best_single: {:.4f}\n"
            "  reports_dir: {}".format(
                best_single_feature["method_id"],
                float(best_single_feature["cv_mae"]),
                best_model.get("model_name", ""),
                float(best_model.get("cv_mae", 0.0)),
                float(best_model.get("cv_mae", 0.0)) - float(best_single_feature["cv_mae"]) if best_model else 0.0,
                reports_dir,
            )
        )

    return {
        "study_summary_json": str(summary_path),
        "fused_features_csv": str(fused_features_path),
        "single_feature_metrics_csv": str(reports_dir / "single_feature_cv_metrics.csv"),
        "model_metrics_csv": str(reports_dir / "depth_only_model_metrics.csv"),
        "best_single_feature_method_id": best_single_feature["method_id"],
        "best_model_name": best_model.get("model_name", ""),
        "sklearn_available": dependencies.get("sklearn", False),
        "xgboost_available": dependencies.get("xgboost", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config YAML.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args()
    result = run_depth_only_model_study(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
