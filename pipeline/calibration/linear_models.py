"""
Linear baselines for the first calibration study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from pipeline.calibration.features import CONTINUOUS_FEATURE_NAMES
from pipeline.calibration.scale_calibration import compute_scale_factor


GEOMETRY_FEATURE_NAMES = [
    "bbox_width_px",
    "bbox_height_px",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
]


def _get_record_value(record: Any, name: str) -> Any:
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, dict):
        return record[name]
    raise TypeError("Record must expose '{}' as an attribute or dict key.".format(name))


@dataclass
class ScaleOnlyModel:
    """Through-origin scale model."""

    scale_factor: float
    model_name: str = "scale_only"

    def predict(self, records: Sequence[Any]) -> np.ndarray:
        values = np.asarray(
            [_get_record_value(record, "relative_depth") for record in records],
            dtype=np.float64,
        )
        return self.scale_factor * values

    def coefficient_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_name": self.model_name,
                "term": "intercept",
                "coefficient": 0.0,
            },
            {
                "model_name": self.model_name,
                "term": "relative_depth",
                "coefficient": float(self.scale_factor),
            },
        ]


@dataclass
class LinearRegressionModel:
    """OLS model whose stored coefficients are already on the original feature scale."""

    model_name: str
    intercept: float
    coefficients: Dict[str, float]
    continuous_feature_names: List[str] = field(default_factory=list)
    categorical_levels: Dict[str, List[str]] = field(default_factory=dict)

    def predict(self, records: Sequence[Any]) -> np.ndarray:
        predictions: List[float] = []
        for record in records:
            value = float(self.intercept)
            for feature_name in self.continuous_feature_names:
                value += float(self.coefficients.get(feature_name, 0.0)) * float(
                    _get_record_value(record, feature_name)
                )
            for feature_name, encoded_levels in self.categorical_levels.items():
                record_value = str(_get_record_value(record, feature_name))
                for level in encoded_levels:
                    term_name = "{}={}".format(feature_name, level)
                    if record_value == level:
                        value += float(self.coefficients.get(term_name, 0.0))
            predictions.append(value)
        return np.asarray(predictions, dtype=np.float64)

    def coefficient_rows(self) -> List[Dict[str, Any]]:
        rows = [
            {
                "model_name": self.model_name,
                "term": "intercept",
                "coefficient": float(self.intercept),
            }
        ]
        for term_name in sorted(self.coefficients):
            rows.append(
                {
                    "model_name": self.model_name,
                    "term": term_name,
                    "coefficient": float(self.coefficients[term_name]),
                }
            )
        return rows


def _extract_targets(records: Sequence[Any]) -> np.ndarray:
    targets = np.asarray(
        [_get_record_value(record, "true_distance_m") for record in records],
        dtype=np.float64,
    )
    if targets.size == 0:
        raise ValueError("Cannot fit a model with no records.")
    return targets


def _fit_linear_model(
    records: Sequence[Any],
    model_name: str,
    continuous_feature_names: Iterable[str],
    categorical_feature_names: Iterable[str],
) -> LinearRegressionModel:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot fit a linear model with no records.")

    y = _extract_targets(rows)
    continuous_feature_names = list(continuous_feature_names)
    categorical_feature_names = list(categorical_feature_names)

    standardized_columns: List[np.ndarray] = []
    original_coefficients: Dict[str, float] = {}
    continuous_means: Dict[str, float] = {}
    continuous_scales: Dict[str, float] = {}

    for feature_name in continuous_feature_names:
        column = np.asarray(
            [_get_record_value(record, feature_name) for record in rows],
            dtype=np.float64,
        )
        mean_value = float(np.mean(column))
        scale_value = float(np.std(column))
        if scale_value == 0.0:
            scale_value = 1.0
        continuous_means[feature_name] = mean_value
        continuous_scales[feature_name] = scale_value
        standardized_columns.append((column - mean_value) / scale_value)

    categorical_levels: Dict[str, List[str]] = {}
    categorical_columns: List[np.ndarray] = []
    categorical_term_names: List[str] = []
    for feature_name in categorical_feature_names:
        levels = sorted({str(_get_record_value(record, feature_name)) for record in rows})
        encoded_levels = levels[1:]
        categorical_levels[feature_name] = encoded_levels
        for level in encoded_levels:
            categorical_columns.append(
                np.asarray(
                    [
                        1.0 if str(_get_record_value(record, feature_name)) == level else 0.0
                        for record in rows
                    ],
                    dtype=np.float64,
                )
            )
            categorical_term_names.append("{}={}".format(feature_name, level))

    design_columns = standardized_columns + categorical_columns
    if design_columns:
        X = np.column_stack(design_columns)
    else:
        X = np.empty((len(rows), 0), dtype=np.float64)

    X_with_intercept = np.column_stack([np.ones(len(rows), dtype=np.float64), X])
    beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    intercept = float(beta[0])
    continuous_beta = beta[1 : 1 + len(continuous_feature_names)]
    categorical_beta = beta[1 + len(continuous_feature_names) :]

    for index, feature_name in enumerate(continuous_feature_names):
        scale_value = continuous_scales[feature_name]
        mean_value = continuous_means[feature_name]
        original_coefficient = float(continuous_beta[index] / scale_value)
        original_coefficients[feature_name] = original_coefficient
        intercept -= float(continuous_beta[index] * mean_value / scale_value)

    for index, term_name in enumerate(categorical_term_names):
        original_coefficients[term_name] = float(categorical_beta[index])

    return LinearRegressionModel(
        model_name=model_name,
        intercept=float(intercept),
        coefficients=original_coefficients,
        continuous_feature_names=continuous_feature_names,
        categorical_levels=categorical_levels,
    )


def fit_scale_only_model(records: Sequence[Any]) -> ScaleOnlyModel:
    """Fit the through-origin global scale baseline."""
    scale_factor = compute_scale_factor(
        relative_depths=[_get_record_value(record, "relative_depth") for record in records],
        true_distances=[_get_record_value(record, "true_distance_m") for record in records],
    )
    return ScaleOnlyModel(scale_factor=scale_factor)


def fit_depth_only_linear_model(records: Sequence[Any]) -> LinearRegressionModel:
    """Fit OLS with an intercept on relative depth only."""
    return _fit_linear_model(
        records=records,
        model_name="depth_only_linear",
        continuous_feature_names=["relative_depth"],
        categorical_feature_names=[],
    )


def fit_depth_geometry_metadata_linear_model(records: Sequence[Any]) -> LinearRegressionModel:
    """Fit OLS with depth, bbox geometry, and one-hot metadata features."""
    return _fit_linear_model(
        records=records,
        model_name="depth_geometry_metadata_linear",
        continuous_feature_names=["relative_depth"] + GEOMETRY_FEATURE_NAMES,
        categorical_feature_names=["weather", "time_of_day"],
    )
