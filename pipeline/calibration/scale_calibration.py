"""
Scale calibration using least squares regression.

Implements the calibration procedure described in the paper:
Z_true ≈ s * d_rel
"""

import numpy as np


def compute_scale_factor(relative_depths, true_distances):
    """
    Compute the scale factor s using least squares regression
    through the origin.

    Parameters
    ----------
    relative_depths
        Sequence of relative depth values (d_rel).
    true_distances
        Sequence of ground-truth distances (Z_true) in meters.

    Returns
    -------
    float
        Scale factor s.
    """
    d = np.asarray(relative_depths, dtype=np.float32)
    z = np.asarray(true_distances, dtype=np.float32)

    if d.size == 0 or z.size == 0:
        raise ValueError("Calibration data is empty.")

    if d.shape[0] != z.shape[0]:
        raise ValueError("Mismatched calibration data lengths.")

    # Least squares solution for Z ≈ s * d
    numerator = np.sum(d * z)
    denominator = np.sum(d * d)

    if denominator == 0.0:
        raise ValueError("Relative depth values are all zero.")

    s = float(numerator / denominator)

    return s
