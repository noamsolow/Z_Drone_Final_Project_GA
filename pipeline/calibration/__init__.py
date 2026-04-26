"""
Calibration and calibration-study utilities.
"""

from pipeline.calibration.evaluation import evaluate_predictions
from pipeline.calibration.features import RegressionRecord, build_regression_dataset
from pipeline.calibration.linear_models import (
    fit_depth_geometry_metadata_linear_model,
    fit_depth_only_linear_model,
    fit_scale_only_model,
)
from pipeline.calibration.scale_calibration import compute_scale_factor
from pipeline.calibration.splits import (
    make_distance_range_label,
    make_fixed_holdout_and_cv_splits,
)

__all__ = [
    "RegressionRecord",
    "build_regression_dataset",
    "compute_scale_factor",
    "evaluate_predictions",
    "fit_depth_geometry_metadata_linear_model",
    "fit_depth_only_linear_model",
    "fit_scale_only_model",
    "make_distance_range_label",
    "make_fixed_holdout_and_cv_splits",
]
