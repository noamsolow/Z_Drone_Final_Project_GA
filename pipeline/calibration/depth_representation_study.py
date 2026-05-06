"""
Evaluation helpers for the second-attempt depth-representation study.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.calibration.depth_representation_features import DEPTH_SCORE_FIELD_NAMES
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions


def _fit_univariate_linear_model(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float]:
    """Fit ``y ~= intercept + slope * x`` with a robust constant-feature fallback."""
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have the same length.")
    if x_values.size == 0:
        return 0.0, 0.0
    if x_values.size == 1 or np.std(x_values) <= 1e-12:
        return float(np.mean(y_values)), 0.0

    design_matrix = np.column_stack([np.ones_like(x_values), x_values])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y_values, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _predict_with_linear_model(
    x_values: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return intercept + slope * np.asarray(x_values, dtype=np.float64)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with stable tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and np.isclose(values[order[end]], values[order[start]]):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _pearson_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size == 0 or np.std(x_values) <= 1e-12 or np.std(y_values) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _spearman_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if len(x_values) == 0:
        return 0.0
    return _pearson_correlation(_average_ranks(x_values), _average_ranks(y_values))


def build_method_id(context_window: str, aggregation_method: str, score_field: str) -> str:
    """Build a stable identifier for one representation method."""
    return "{}__{}__{}".format(context_window, aggregation_method, score_field)


def evaluate_depth_representation_methods(
    records: Sequence[Any],
    score_fields: Sequence[str] = DEPTH_SCORE_FIELD_NAMES,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Evaluate every ``context window x aggregation x score field`` combination.

    Each candidate method is evaluated with leave-one-out linear calibration.
    """
    rows = list(records)
    if not rows:
        raise ValueError("At least one depth representation record is required.")

    for score_field in score_fields:
        if score_field not in DEPTH_SCORE_FIELD_NAMES:
            raise ValueError("Unsupported score_field '{}'.".format(score_field))

    grouped_records: Dict[Tuple[str, str], List[Any]] = {}
    for record in rows:
        key = (str(record.context_window), str(record.aggregation_method))
        grouped_records.setdefault(key, []).append(record)

    method_metric_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []
    total_methods = len(grouped_records) * len(score_fields)
    method_counter = 0

    for (context_window, aggregation_method), method_records in sorted(grouped_records.items()):
        method_records = sorted(
            method_records,
            key=lambda record: (
                float(record.true_distance_m),
                str(record.weather),
                str(record.time_of_day),
                str(record.image_path),
            ),
        )
        true_values = np.asarray(
            [float(record.true_distance_m) for record in method_records],
            dtype=np.float64,
        )

        for score_field in score_fields:
            method_counter += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "method_start",
                        "method_index": method_counter,
                        "total_methods": total_methods,
                        "context_window": context_window,
                        "aggregation_method": aggregation_method,
                        "score_field": score_field,
                    }
                )
            x_values = np.asarray(
                [float(getattr(record, score_field)) for record in method_records],
                dtype=np.float64,
            )
            predictions = np.zeros_like(true_values, dtype=np.float64)
            for left_out_index in range(len(method_records)):
                train_mask = np.ones(len(method_records), dtype=bool)
                train_mask[left_out_index] = False
                intercept, slope = _fit_univariate_linear_model(
                    x_values[train_mask],
                    true_values[train_mask],
                )
                predictions[left_out_index] = _predict_with_linear_model(
                    x_values[left_out_index:left_out_index + 1],
                    intercept=intercept,
                    slope=slope,
                )[0]

            evaluation = evaluate_predictions(method_records, predictions)
            method_id = build_method_id(context_window, aggregation_method, score_field)
            pearson_correlation = _pearson_correlation(x_values, true_values)
            spearman_correlation = _spearman_correlation(x_values, true_values)
            full_intercept, full_slope = _fit_univariate_linear_model(x_values, true_values)

            method_metric_rows.append(
                {
                    "method_id": method_id,
                    "context_window": context_window,
                    "aggregation_method": aggregation_method,
                    "score_field": score_field,
                    "count": evaluation["overall"]["count"],
                    "pearson_correlation": pearson_correlation,
                    "spearman_correlation": spearman_correlation,
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
                    "full_fit_intercept": full_intercept,
                    "full_fit_slope": full_slope,
                }
            )
            coefficient_rows.append(
                {
                    "method_id": method_id,
                    "context_window": context_window,
                    "aggregation_method": aggregation_method,
                    "score_field": score_field,
                    "full_fit_intercept": full_intercept,
                    "full_fit_slope": full_slope,
                }
            )
            grouped_summary_rows.extend(
                build_grouped_error_summary(
                    records=method_records,
                    predictions=predictions,
                    model_name=method_id,
                    split_name="leave_one_out",
                )
            )

            for record, prediction, score_value in zip(method_records, predictions, x_values):
                prediction_rows.append(
                    {
                        "method_id": method_id,
                        "context_window": context_window,
                        "aggregation_method": aggregation_method,
                        "score_field": score_field,
                        "image_path": record.image_path,
                        "label_path": record.label_path,
                        "stratum_key": record.stratum_key,
                        "selection_rank_within_stratum": int(record.selection_rank_within_stratum),
                        "weather": record.weather,
                        "time_of_day": record.time_of_day,
                        "true_distance_m": float(record.true_distance_m),
                        "score_value": float(score_value),
                        "predicted_distance_m": float(prediction),
                        "signed_error_m": float(prediction) - float(record.true_distance_m),
                        "absolute_error_m": abs(float(prediction) - float(record.true_distance_m)),
                        "relative_error": abs(float(prediction) - float(record.true_distance_m))
                        / float(record.true_distance_m),
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "method_complete",
                        "method_index": method_counter,
                        "total_methods": total_methods,
                        "method_id": method_id,
                        "loocv_mae": evaluation["overall"]["mae"],
                        "loocv_mean_relative_error": evaluation["overall"]["mean_relative_error"],
                        "loocv_r2": evaluation["overall"]["r2"],
                    }
                )

    ranking = sorted(
        method_metric_rows,
        key=lambda row: (
            row["loocv_mae"],
            row["loocv_mean_relative_error"],
            -row["spearman_correlation"],
            row["method_id"],
        ),
    )

    for rank_index, row in enumerate(ranking, start=1):
        row["rank"] = rank_index

    winner = dict(ranking[0]) if ranking else {}
    if ranking:
        winner_mae = float(ranking[0]["loocv_mae"])
        for row in ranking:
            row["mae_gap_from_winner"] = float(row["loocv_mae"]) - winner_mae

    best_by_context = {
        context: dict(rows[0])
        for context, rows in sorted(
            {
                context: sorted(
                    [row for row in ranking if row["context_window"] == context],
                    key=lambda row: (
                        row["loocv_mae"],
                        row["loocv_mean_relative_error"],
                        -row["spearman_correlation"],
                        row["method_id"],
                    ),
                )
                for context in {row["context_window"] for row in ranking}
            }.items()
        )
    }
    best_by_aggregation = {
        aggregation: dict(rows[0])
        for aggregation, rows in sorted(
            {
                aggregation: sorted(
                    [row for row in ranking if row["aggregation_method"] == aggregation],
                    key=lambda row: (
                        row["loocv_mae"],
                        row["loocv_mean_relative_error"],
                        -row["spearman_correlation"],
                        row["method_id"],
                    ),
                )
                for aggregation in {row["aggregation_method"] for row in ranking}
            }.items()
        )
    }
    best_by_score_field = {
        score_field: dict(rows[0])
        for score_field, rows in sorted(
            {
                score_field: sorted(
                    [row for row in ranking if row["score_field"] == score_field],
                    key=lambda row: (
                        row["loocv_mae"],
                        row["loocv_mean_relative_error"],
                        -row["spearman_correlation"],
                        row["method_id"],
                    ),
                )
                for score_field in {row["score_field"] for row in ranking}
            }.items()
        )
    }

    return {
        "winner": winner,
        "ranking": ranking,
        "method_metric_rows": method_metric_rows,
        "coefficient_rows": coefficient_rows,
        "prediction_rows": prediction_rows,
        "grouped_summary_rows": grouped_summary_rows,
        "best_by_context": best_by_context,
        "best_by_aggregation": best_by_aggregation,
        "best_by_score_field": best_by_score_field,
    }
