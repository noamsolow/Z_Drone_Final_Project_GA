"""
Evaluation helpers for the first calibration study.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from pipeline.calibration.features import CONTINUOUS_FEATURE_NAMES
from pipeline.calibration.splits import make_distance_range_label


def _get_record_value(record: Any, name: str) -> Any:
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, dict):
        return record[name]
    raise TypeError("Record must expose '{}' as an attribute or dict key.".format(name))


def _metric_dict(true_values: np.ndarray, predicted_values: np.ndarray) -> Dict[str, float]:
    if true_values.size == 0:
        return {
            "count": 0,
            "mae": 0.0,
            "median_absolute_error": 0.0,
            "mean_relative_error": 0.0,
            "median_relative_error": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "mean_signed_error": 0.0,
            "median_signed_error": 0.0,
            "p90_absolute_error": 0.0,
            "p95_absolute_error": 0.0,
            "max_absolute_error": 0.0,
            "within_5m_rate": 0.0,
            "within_10m_rate": 0.0,
            "within_20m_rate": 0.0,
            "overprediction_rate": 0.0,
            "underprediction_rate": 0.0,
        }

    signed_errors = predicted_values - true_values
    absolute_errors = np.abs(predicted_values - true_values)
    mae = float(np.mean(absolute_errors))
    relative_errors = absolute_errors / np.abs(true_values)
    rmse = float(np.sqrt(np.mean((predicted_values - true_values) ** 2)))
    ss_res = float(np.sum((predicted_values - true_values) ** 2))
    ss_tot = float(np.sum((true_values - np.mean(true_values)) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    return {
        "count": int(true_values.size),
        "mae": mae,
        "median_absolute_error": float(np.median(absolute_errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "median_relative_error": float(np.median(relative_errors)),
        "rmse": rmse,
        "r2": r2,
        "mean_signed_error": float(np.mean(signed_errors)),
        "median_signed_error": float(np.median(signed_errors)),
        "p90_absolute_error": float(np.percentile(absolute_errors, 90)),
        "p95_absolute_error": float(np.percentile(absolute_errors, 95)),
        "max_absolute_error": float(np.max(absolute_errors)),
        "within_5m_rate": float(np.mean(absolute_errors <= 5.0)),
        "within_10m_rate": float(np.mean(absolute_errors <= 10.0)),
        "within_20m_rate": float(np.mean(absolute_errors <= 20.0)),
        "overprediction_rate": float(np.mean(signed_errors > 0.0)),
        "underprediction_rate": float(np.mean(signed_errors < 0.0)),
    }


def evaluate_predictions(records: Sequence[Any], predictions: Sequence[float]) -> Dict[str, Any]:
    """Evaluate predictions overall and by distance range."""
    rows = list(records)
    predicted_values = np.asarray(predictions, dtype=np.float64)
    if len(rows) != predicted_values.size:
        raise ValueError("records and predictions must have the same length.")

    true_values = np.asarray(
        [_get_record_value(record, "true_distance_m") for record in rows],
        dtype=np.float64,
    )
    evaluation = {
        "overall": _metric_dict(true_values, predicted_values),
        "per_distance_range": {},
    }

    grouped_true: Dict[str, List[float]] = defaultdict(list)
    grouped_pred: Dict[str, List[float]] = defaultdict(list)
    for record, prediction in zip(rows, predicted_values):
        range_label = make_distance_range_label(_get_record_value(record, "true_distance_m"))
        grouped_true[range_label].append(float(_get_record_value(record, "true_distance_m")))
        grouped_pred[range_label].append(float(prediction))

    for range_label in ("near", "mid", "far"):
        evaluation["per_distance_range"][range_label] = _metric_dict(
            np.asarray(grouped_true.get(range_label, []), dtype=np.float64),
            np.asarray(grouped_pred.get(range_label, []), dtype=np.float64),
        )

    return evaluation


def compute_feature_correlations(records: Iterable[Any]) -> List[Dict[str, Any]]:
    """Compute Pearson correlation between each continuous feature and ground truth."""
    rows = list(records)
    true_values = np.asarray(
        [_get_record_value(record, "true_distance_m") for record in rows],
        dtype=np.float64,
    )

    output_rows: List[Dict[str, Any]] = []
    for feature_name in CONTINUOUS_FEATURE_NAMES:
        feature_values = np.asarray(
            [_get_record_value(record, feature_name) for record in rows],
            dtype=np.float64,
        )
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

    return output_rows


def build_grouped_error_summary(
    records: Sequence[Any],
    predictions: Sequence[float],
    model_name: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    """Build grouped summary rows for metadata and distance-range slices."""
    rows = list(records)
    predicted_values = np.asarray(predictions, dtype=np.float64)
    if len(rows) != predicted_values.size:
        raise ValueError("records and predictions must have the same length.")

    grouping_to_records: Dict[str, Dict[str, List[int]]] = {
        "weather": defaultdict(list),
        "time_of_day": defaultdict(list),
        "distance_range": defaultdict(list),
        "weather__distance_range": defaultdict(list),
        "time_of_day__distance_range": defaultdict(list),
    }

    for index, record in enumerate(rows):
        weather = str(_get_record_value(record, "weather"))
        time_of_day = str(_get_record_value(record, "time_of_day"))
        distance_range = make_distance_range_label(_get_record_value(record, "true_distance_m"))
        grouping_to_records["weather"][weather].append(index)
        grouping_to_records["time_of_day"][time_of_day].append(index)
        grouping_to_records["distance_range"][distance_range].append(index)
        grouping_to_records["weather__distance_range"]["{}|{}".format(weather, distance_range)].append(
            index
        )
        grouping_to_records["time_of_day__distance_range"][
            "{}|{}".format(time_of_day, distance_range)
        ].append(index)

    summary_rows: List[Dict[str, Any]] = []
    for grouping, group_map in grouping_to_records.items():
        for group_key, group_indices in sorted(group_map.items()):
            group_true = np.asarray(
                [_get_record_value(rows[index], "true_distance_m") for index in group_indices],
                dtype=np.float64,
            )
            group_pred = predicted_values[group_indices]
            metrics = _metric_dict(group_true, group_pred)

            weather = ""
            time_of_day = ""
            distance_range = ""
            if grouping == "weather":
                weather = group_key
            elif grouping == "time_of_day":
                time_of_day = group_key
            elif grouping == "distance_range":
                distance_range = group_key
            elif grouping == "weather__distance_range":
                weather, distance_range = group_key.split("|", 1)
            elif grouping == "time_of_day__distance_range":
                time_of_day, distance_range = group_key.split("|", 1)

            summary_rows.append(
                {
                    "model_name": model_name,
                    "split_name": split_name,
                    "grouping": grouping,
                    "weather": weather,
                    "time_of_day": time_of_day,
                    "distance_range": distance_range,
                    "count": metrics["count"],
                    "mean_true_distance_m": float(np.mean(group_true)) if group_true.size else 0.0,
                    "std_true_distance_m": float(np.std(group_true)) if group_true.size else 0.0,
                    "mean_prediction_m": float(np.mean(group_pred)) if group_pred.size else 0.0,
                    "mae": metrics["mae"],
                    "median_absolute_error": metrics["median_absolute_error"],
                    "mean_relative_error": metrics["mean_relative_error"],
                    "median_relative_error": metrics["median_relative_error"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mean_signed_error": metrics["mean_signed_error"],
                    "median_signed_error": metrics["median_signed_error"],
                    "p90_absolute_error": metrics["p90_absolute_error"],
                    "p95_absolute_error": metrics["p95_absolute_error"],
                    "max_absolute_error": metrics["max_absolute_error"],
                    "within_5m_rate": metrics["within_5m_rate"],
                    "within_10m_rate": metrics["within_10m_rate"],
                    "within_20m_rate": metrics["within_20m_rate"],
                    "overprediction_rate": metrics["overprediction_rate"],
                    "underprediction_rate": metrics["underprediction_rate"],
                }
            )

    return summary_rows
