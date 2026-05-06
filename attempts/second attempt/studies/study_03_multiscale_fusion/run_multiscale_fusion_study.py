"""
Run the Study 03 multiscale fusion search on the cached per-context features.
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

from pipeline.calibration.depth_representation_features import (
    DEPTH_SCORE_FIELD_NAMES,
    read_depth_representation_records_csv,
)
from pipeline.calibration.depth_representation_study import build_method_id
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv_rows(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    x_train_standardized = (x_train - means) / scales
    return x_train_standardized, means, scales


def _fit_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    ridge_alpha: float = 0.0,
) -> Dict[str, Any]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D.")
    if x_train.shape[0] != y_train.size:
        raise ValueError("x_train and y_train must have matching rows.")
    if x_train.shape[0] == 0:
        raise ValueError("Cannot fit on an empty training set.")

    x_train_standardized, means, scales = _standardize_train_matrix(x_train)
    design = np.column_stack([np.ones(x_train_standardized.shape[0], dtype=np.float64), x_train_standardized])

    if model_type == "ols":
        coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    elif model_type == "ridge":
        alpha = float(ridge_alpha)
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        lhs = design.T @ design + alpha * penalty
        rhs = design.T @ y_train
        coefficients = np.linalg.solve(lhs, rhs)
    else:
        raise ValueError("Unsupported model_type '{}'.".format(model_type))

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
    x_values = np.asarray(x_values, dtype=np.float64)
    coefficients = np.asarray(fit_result["coefficients"], dtype=np.float64)
    intercept = float(fit_result["intercept"])
    return intercept + x_values @ coefficients


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

    pivoted_rows = sorted(
        rows_by_image.values(),
        key=lambda row: (
            float(row["true_distance_m"]),
            str(row["weather"]),
            str(row["time_of_day"]),
            str(row["image_path"]),
        ),
    )

    missing_features: List[str] = []
    for row in pivoted_rows:
        for feature_name in feature_names:
            if feature_name not in row:
                missing_features.append(feature_name)
                break
    if missing_features:
        raise ValueError("Pivoted rows are missing feature values for at least one method.")

    return pivoted_rows, feature_names


def _evaluate_feature_set(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    model_name: str,
    model_type: str,
    ridge_alpha: float = 0.0,
) -> Dict[str, Any]:
    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("feature_names must not be empty.")

    x_values = np.asarray(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=np.float64,
    )
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)

    predictions = np.zeros(true_values.shape[0], dtype=np.float64)
    for left_out_index in range(true_values.shape[0]):
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[left_out_index] = False
        fit_result = _fit_linear_regression(
            x_train=x_values[train_mask],
            y_train=true_values[train_mask],
            model_type=model_type,
            ridge_alpha=ridge_alpha,
        )
        predictions[left_out_index] = _predict_linear_regression(
            x_values[left_out_index:left_out_index + 1],
            fit_result,
        )[0]

    full_fit = _fit_linear_regression(
        x_train=x_values,
        y_train=true_values,
        model_type=model_type,
        ridge_alpha=ridge_alpha,
    )
    evaluation = evaluate_predictions(rows, predictions)
    grouped_rows = build_grouped_error_summary(
        records=rows,
        predictions=predictions,
        model_name=model_name,
        split_name="leave_one_out",
    )

    coefficient_rows = [
        {
            "model_name": model_name,
            "model_type": model_type,
            "ridge_alpha": float(ridge_alpha),
            "term": "intercept",
            "coefficient": float(full_fit["intercept"]),
        }
    ]
    for feature_name, coefficient in zip(feature_names, full_fit["coefficients"]):
        coefficient_rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "ridge_alpha": float(ridge_alpha),
                "term": feature_name,
                "coefficient": float(coefficient),
            }
        )

    prediction_rows = []
    for row, prediction in zip(rows, predictions):
        signed_error = float(prediction) - float(row["true_distance_m"])
        prediction_rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "ridge_alpha": float(ridge_alpha),
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

    metric_row = {
        "model_name": model_name,
        "model_type": model_type,
        "ridge_alpha": float(ridge_alpha),
        "feature_set_size": len(feature_names),
        "feature_names": " | ".join(feature_names),
        "count": evaluation["overall"]["count"],
        "loocv_mae": evaluation["overall"]["mae"],
        "loocv_median_absolute_error": evaluation["overall"]["median_absolute_error"],
        "loocv_mean_relative_error": evaluation["overall"]["mean_relative_error"],
        "loocv_median_relative_error": evaluation["overall"]["median_relative_error"],
        "loocv_rmse": evaluation["overall"]["rmse"],
        "loocv_r2": evaluation["overall"]["r2"],
        "loocv_mean_signed_error": evaluation["overall"]["mean_signed_error"],
        "loocv_median_signed_error": evaluation["overall"]["median_signed_error"],
        "loocv_p90_absolute_error": evaluation["overall"]["p90_absolute_error"],
        "loocv_p95_absolute_error": evaluation["overall"]["p95_absolute_error"],
        "loocv_max_absolute_error": evaluation["overall"]["max_absolute_error"],
        "loocv_within_5m_rate": evaluation["overall"]["within_5m_rate"],
        "loocv_within_10m_rate": evaluation["overall"]["within_10m_rate"],
        "loocv_within_20m_rate": evaluation["overall"]["within_20m_rate"],
        "loocv_overprediction_rate": evaluation["overall"]["overprediction_rate"],
        "loocv_underprediction_rate": evaluation["overall"]["underprediction_rate"],
    }

    return {
        "metric_row": metric_row,
        "coefficient_rows": coefficient_rows,
        "prediction_rows": prediction_rows,
        "grouped_rows": grouped_rows,
        "feature_names": feature_names,
    }


def _choose_best_by_context(single_feature_rows: Sequence[Dict[str, Any]]) -> List[str]:
    best_by_context: Dict[str, str] = {}
    for row in single_feature_rows:
        context_window = str(row["context_window"])
        if context_window not in best_by_context:
            best_by_context[context_window] = str(row["method_id"])
    return [best_by_context[key] for key in sorted(best_by_context)]


def _method_id_exists(feature_name: str, feature_name_set: Sequence[str]) -> bool:
    return feature_name in set(feature_name_set)


def _filter_existing_feature_names(feature_names: Iterable[str], available_feature_names: Sequence[str]) -> List[str]:
    available_set = set(available_feature_names)
    return [feature_name for feature_name in feature_names if feature_name in available_set]


def _top_ranked_method_ids(single_feature_rows: Sequence[Dict[str, Any]], count: int) -> List[str]:
    return [str(row["method_id"]) for row in single_feature_rows[:count]]


def _variance_filter(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> List[str]:
    kept: List[str] = []
    for feature_name in feature_names:
        values = np.asarray([float(row[feature_name]) for row in rows], dtype=np.float64)
        if np.std(values) > 1e-12:
            kept.append(feature_name)
    return kept


def _generate_candidate_specs(
    rows: Sequence[Dict[str, Any]],
    available_feature_names: Sequence[str],
    single_feature_rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    study_config = config.get("study", {})
    context_specs = study_config.get("context_windows", [])
    context_order = [str(spec["name"]) for spec in context_specs]

    candidate_specs: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"greedy_paths": {}}

    def add_spec(model_name: str, feature_names: Sequence[str], model_type: str, ridge_alpha: float = 0.0) -> None:
        filtered_features = _variance_filter(rows, _filter_existing_feature_names(feature_names, available_feature_names))
        if not filtered_features:
            return
        candidate_specs.append(
            {
                "model_name": model_name,
                "feature_names": filtered_features,
                "model_type": model_type,
                "ridge_alpha": float(ridge_alpha),
            }
        )

    for top_k in (2, 3, 5, 8, 12):
        top_features = _top_ranked_method_ids(single_feature_rows, top_k)
        add_spec("top_{}_ols".format(top_k), top_features, model_type="ols")
        add_spec("top_{}_ridge_alpha_1".format(top_k), top_features, model_type="ridge", ridge_alpha=1.0)
        add_spec("top_{}_ridge_alpha_10".format(top_k), top_features, model_type="ridge", ridge_alpha=10.0)

    best_per_context = _choose_best_by_context(single_feature_rows)
    add_spec("best_per_context_ols", best_per_context, model_type="ols")
    add_spec("best_per_context_ridge_alpha_1", best_per_context, model_type="ridge", ridge_alpha=1.0)

    raw_midpoint_all_scales = [
        build_method_id(context_name, "bbox_midpoint", "object_depth")
        for context_name in context_order
    ]
    raw_midpoint_core_scales = [
        build_method_id(context_name, "bbox_midpoint", "object_depth")
        for context_name in ("bbox_only", "bbox_expand_1_5x", "bbox_expand_2x", "bbox_expand_4x")
    ]
    contrast_midpoint_scales = [
        build_method_id(context_name, "bbox_midpoint", "object_minus_ring")
        for context_name in context_order
        if context_name != "bbox_only"
    ]
    contrast_midpoint_core = [
        build_method_id(context_name, "bbox_midpoint", "object_minus_ring")
        for context_name in ("bbox_expand_1_5x", "bbox_expand_2x", "bbox_expand_4x", "bbox_expand_6x")
    ]
    normalized_midpoint_all_scales = [
        build_method_id(context_name, "bbox_midpoint", "object_depth_percentile_5_95")
        for context_name in context_order
    ]
    mixed_shortlist = [
        build_method_id("bbox_only", "bbox_midpoint", "object_depth"),
        build_method_id("bbox_expand_1_5x", "bbox_midpoint", "object_minus_ring"),
        build_method_id("bbox_expand_2x", "bbox_midpoint", "object_minus_ring"),
        build_method_id("bbox_expand_4x", "bbox_midpoint", "object_minus_ring"),
    ]

    add_spec("raw_midpoint_all_scales_ols", raw_midpoint_all_scales, model_type="ols")
    add_spec("raw_midpoint_all_scales_ridge_alpha_1", raw_midpoint_all_scales, model_type="ridge", ridge_alpha=1.0)
    add_spec("raw_midpoint_core_scales_ols", raw_midpoint_core_scales, model_type="ols")
    add_spec("raw_midpoint_core_scales_ridge_alpha_1", raw_midpoint_core_scales, model_type="ridge", ridge_alpha=1.0)
    add_spec("contrast_midpoint_all_scales_ols", contrast_midpoint_scales, model_type="ols")
    add_spec("contrast_midpoint_all_scales_ridge_alpha_1", contrast_midpoint_scales, model_type="ridge", ridge_alpha=1.0)
    add_spec("contrast_midpoint_core_scales_ols", contrast_midpoint_core, model_type="ols")
    add_spec("contrast_midpoint_core_scales_ridge_alpha_1", contrast_midpoint_core, model_type="ridge", ridge_alpha=1.0)
    add_spec("normalized_midpoint_all_scales_ols", normalized_midpoint_all_scales, model_type="ols")
    add_spec("normalized_midpoint_all_scales_ridge_alpha_1", normalized_midpoint_all_scales, model_type="ridge", ridge_alpha=1.0)
    add_spec("mixed_shortlist_ols", mixed_shortlist, model_type="ols")
    add_spec("mixed_shortlist_ridge_alpha_1", mixed_shortlist, model_type="ridge", ridge_alpha=1.0)

    top_pool = _top_ranked_method_ids(single_feature_rows, 20)
    for model_type, ridge_alpha in (("ols", 0.0), ("ridge", 1.0)):
        selected: List[str] = []
        path_rows: List[Dict[str, Any]] = []
        best_previous_mae: Optional[float] = None
        for step_index in range(1, 7):
            best_candidate_result = None
            best_candidate_feature = None
            for candidate_feature in top_pool:
                if candidate_feature in selected:
                    continue
                candidate_features = selected + [candidate_feature]
                result = _evaluate_feature_set(
                    rows=rows,
                    feature_names=candidate_features,
                    model_name="greedy_search_temp",
                    model_type=model_type,
                    ridge_alpha=ridge_alpha,
                )
                candidate_mae = float(result["metric_row"]["loocv_mae"])
                if best_candidate_result is None or candidate_mae < float(best_candidate_result["metric_row"]["loocv_mae"]):
                    best_candidate_result = result
                    best_candidate_feature = candidate_feature
            if best_candidate_result is None or best_candidate_feature is None:
                break
            candidate_mae = float(best_candidate_result["metric_row"]["loocv_mae"])
            if best_previous_mae is not None and candidate_mae >= best_previous_mae - 1e-6:
                break
            selected.append(best_candidate_feature)
            best_previous_mae = candidate_mae
            model_name = "greedy_forward_{}_k{}".format(model_type, len(selected))
            add_spec(model_name, list(selected), model_type=model_type, ridge_alpha=ridge_alpha)
            path_rows.append(
                {
                    "step": len(selected),
                    "selected_feature": best_candidate_feature,
                    "loocv_mae": candidate_mae,
                    "feature_names": list(selected),
                }
            )
        diagnostics["greedy_paths"]["{}_alpha_{}".format(model_type, ridge_alpha)] = path_rows

    deduped_specs: List[Dict[str, Any]] = []
    seen_keys = set()
    for spec in candidate_specs:
        dedupe_key = (
            tuple(spec["feature_names"]),
            spec["model_type"],
            float(spec["ridge_alpha"]),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_specs.append(spec)

    return deduped_specs, diagnostics


def run_multiscale_fusion_study(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_csv_path = output_root / "features" / "representation_records.csv"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    single_feature_metrics_path = reports_dir / "method_metrics.csv"
    if not features_csv_path.exists():
        raise FileNotFoundError("Expected Study 03 features CSV at {}.".format(features_csv_path))
    if not single_feature_metrics_path.exists():
        raise FileNotFoundError(
            "Expected single-scale ranking at {}. Run attempts/second attempt/run_study.py "
            "with the Study 03 config first.".format(single_feature_metrics_path)
        )

    records = read_depth_representation_records_csv(features_csv_path)
    image_rows, available_feature_names = _pivot_records_to_image_rows(records)
    single_feature_rows = _read_csv_rows(single_feature_metrics_path)
    single_feature_rows = sorted(single_feature_rows, key=lambda row: int(row["rank"]))

    if verbose:
        print(
            "[study-03] Starting multiscale fusion study\n"
            "  config: {}\n"
            "  output_root: {}\n"
            "  features_csv: {}\n"
            "  single_feature_metrics_csv: {}\n"
            "  num_image_rows: {}\n"
            "  num_available_features: {}".format(
                config_path or DEFAULT_CONFIG_PATH,
                output_root,
                features_csv_path,
                single_feature_metrics_path,
                len(image_rows),
                len(available_feature_names),
            )
        )

    candidate_specs, diagnostics = _generate_candidate_specs(
        rows=image_rows,
        available_feature_names=available_feature_names,
        single_feature_rows=single_feature_rows,
        config=resolved_config,
    )

    if verbose:
        print(
            "[study-03] Candidate fusion models prepared\n"
            "  num_candidates: {}\n"
            "  top_single_feature: {}\n"
            "  best_single_feature_mae: {:.4f}".format(
                len(candidate_specs),
                single_feature_rows[0]["method_id"],
                float(single_feature_rows[0]["loocv_mae"]),
            )
        )

    metric_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    grouped_rows: List[Dict[str, Any]] = []

    for candidate_index, candidate_spec in enumerate(candidate_specs, start=1):
        if verbose:
            print(
                "[study-03] Evaluating candidate {}/{}\n"
                "  model_name: {}\n"
                "  model_type: {}\n"
                "  ridge_alpha: {}\n"
                "  feature_count: {}".format(
                    candidate_index,
                    len(candidate_specs),
                    candidate_spec["model_name"],
                    candidate_spec["model_type"],
                    candidate_spec["ridge_alpha"],
                    len(candidate_spec["feature_names"]),
                )
            )
        result = _evaluate_feature_set(
            rows=image_rows,
            feature_names=candidate_spec["feature_names"],
            model_name=candidate_spec["model_name"],
            model_type=candidate_spec["model_type"],
            ridge_alpha=float(candidate_spec["ridge_alpha"]),
        )
        metric_rows.append(result["metric_row"])
        coefficient_rows.extend(result["coefficient_rows"])
        prediction_rows.extend(result["prediction_rows"])
        grouped_rows.extend(result["grouped_rows"])

    ranking = sorted(
        metric_rows,
        key=lambda row: (
            float(row["loocv_mae"]),
            float(row["loocv_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for rank_index, row in enumerate(ranking, start=1):
        row["rank"] = rank_index

    best_single_feature = dict(single_feature_rows[0])
    best_fusion_model = dict(ranking[0]) if ranking else {}
    best_single_mae = float(best_single_feature["loocv_mae"])
    if ranking:
        for row in ranking:
            row["mae_gap_from_best_fusion"] = float(row["loocv_mae"]) - float(best_fusion_model["loocv_mae"])
            row["mae_gap_vs_best_single"] = float(row["loocv_mae"]) - best_single_mae

    summary = {
        "study_name": resolved_config.get("study_name", output_root.name),
        "best_single_feature": best_single_feature,
        "best_fusion_model": best_fusion_model,
        "gap_vs_best_single_feature": {
            "loocv_mae": float(best_fusion_model.get("loocv_mae", 0.0)) - best_single_mae if best_fusion_model else 0.0,
            "loocv_mean_relative_error": float(best_fusion_model.get("loocv_mean_relative_error", 0.0))
            - float(best_single_feature["loocv_mean_relative_error"])
            if best_fusion_model
            else 0.0,
            "loocv_rmse": float(best_fusion_model.get("loocv_rmse", 0.0)) - float(best_single_feature["loocv_rmse"])
            if best_fusion_model
            else 0.0,
        },
        "num_image_rows": len(image_rows),
        "num_available_features": len(available_feature_names),
        "num_fusion_models_tested": len(ranking),
        "top_10_single_features": single_feature_rows[:10],
        "top_10_fusion_models": ranking[:10],
        "greedy_search_diagnostics": diagnostics["greedy_paths"],
    }
    summary_path = reports_dir / "fusion_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_csv_rows(ranking, reports_dir / "fusion_model_metrics.csv")
    _write_csv_rows(coefficient_rows, reports_dir / "fusion_model_coefficients.csv")
    _write_csv_rows(prediction_rows, reports_dir / "fusion_loo_predictions.csv")
    _write_csv_rows(grouped_rows, reports_dir / "fusion_grouped_error_summary.csv")
    _write_csv_rows(
        [row for row in grouped_rows if row.get("grouping") == "distance_range"],
        reports_dir / "fusion_distance_range_metrics.csv",
    )

    if verbose and best_fusion_model:
        print(
            "[study-03] Multiscale fusion study complete\n"
            "  best_single_feature: {}\n"
            "  best_single_feature_mae: {:.4f}\n"
            "  best_fusion_model: {}\n"
            "  best_fusion_model_mae: {:.4f}\n"
            "  mae_delta_vs_best_single: {:.4f}\n"
            "  reports_dir: {}".format(
                best_single_feature["method_id"],
                float(best_single_feature["loocv_mae"]),
                best_fusion_model["model_name"],
                float(best_fusion_model["loocv_mae"]),
                float(best_fusion_model["loocv_mae"]) - best_single_mae,
                reports_dir,
            )
        )

    return {
        "fusion_study_summary_json": str(summary_path),
        "fusion_model_metrics_csv": str(reports_dir / "fusion_model_metrics.csv"),
        "best_single_feature_method_id": best_single_feature["method_id"],
        "best_fusion_model_name": best_fusion_model.get("model_name", ""),
        "num_fusion_models_tested": len(ranking),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output and print only the final JSON result.",
    )
    args = parser.parse_args()
    result = run_multiscale_fusion_study(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
