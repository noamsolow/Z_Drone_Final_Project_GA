"""
Monocular drone localization pipeline.

Implements the full workflow described in the paper:
- Calibration mode (offline): estimate scale factor s
- Runtime mode (inference): estimate 3D drone position
"""

from pipeline.detection.detector import detect_drone, detect_drone_demo
from pipeline.depth.depth_model import (
    estimate_relative_depth,
    estimate_relative_depth_demo,
)
from pipeline.depth.drone_depth import extract_drone_relative_depth
from pipeline.calibration.scale_calibration import compute_scale_factor
from pipeline.geometry.projection import reconstruct_drone_position


# ============================================================
# CALIBRATION MODE (OFFLINE)
# ============================================================

def run_calibration_pipeline( images, true_distances, detection_mode="demo", depth_mode="demo", detector_model=None, depth_model=None,):
    """
    Run the calibration pipeline to estimate the scale factor s.

    Parameters
    ----------
    images
        Iterable of RGB images.
    true_distances
        Iterable of ground-truth distances (Z_true) in meters.
    detection_mode
        "demo" or "real".
    depth_mode
        "demo" or "real".
    detector_model
        Loaded detector model (required if detection_mode == "real").
    depth_model
        Loaded depth model (required if depth_mode == "real").

    Returns
    -------
    float
        Estimated scale factor s.
    """
    relative_depths = []
    z_true_values = []

    for image, z_true in zip(images, true_distances):

        # ---- Detection ----
        if detection_mode == "demo":
            detection = detect_drone_demo(image)
        else:
            detection = detect_drone(image, detector_model)

        bbox = detection["bbox"]

        # ---- Depth estimation ----
        if depth_mode == "demo":
            depth_map = estimate_relative_depth_demo(image)
        else:
            depth_map = estimate_relative_depth(image, depth_model)

        # ---- Relative depth at drone location ----
        d_rel = extract_drone_relative_depth(depth_map, bbox)

        relative_depths.append(d_rel)
        z_true_values.append(z_true)

    # ---- Scale calibration ----
    s = compute_scale_factor(relative_depths, z_true_values)

    return s


# ============================================================
# RUNTIME MODE (INFERENCE)
# ============================================================

def run_runtime_pipeline(image, scale_factor, intrinsics, detection_mode="demo", depth_mode="demo", detector_model=None, depth_model=None,):
    """
    Run the runtime pipeline to estimate the 3D position of the drone.

    Parameters
    ----------
    image
        Input RGB image.
    scale_factor
        Precomputed scale factor s.
    intrinsics
        Camera intrinsic parameters:
        {
            "fx": float,
            "fy": float,
            "cx": float,
            "cy": float
        }
    detection_mode
        "demo" or "real".
    depth_mode
        "demo" or "real".
    detector_model
        Loaded detector model (required if detection_mode == "real").
    depth_model
        Loaded depth model (required if depth_mode == "real").

    Returns
    -------
    tuple
        Estimated 3D position of the drone (X, Y, Z).
    """

    # ---- Detection ----
    if detection_mode == "demo":
        detection = detect_drone_demo(image)
    else:
        detection = detect_drone(image, detector_model)

    bbox = detection["bbox"]
    x_pixel = detection["x_pixel"]
    y_pixel = detection["y_pixel"]

    # ---- Depth estimation ----
    if depth_mode == "demo":
        depth_map = estimate_relative_depth_demo(image)
    else:
        depth_map = estimate_relative_depth(image, depth_model)

    # ---- Relative depth at drone location ----
    d_rel = extract_drone_relative_depth(depth_map, bbox)

    # ---- Metric depth ----
    z_hat = scale_factor * d_rel

    # ---- Geometry ----
    position_3d = reconstruct_drone_position(
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        z=z_hat,
        intrinsics=intrinsics,
    )

    return position_3d



if __name__ == "__main__":
    # ---- Fake data ----
    num_samples = 10
    fake_images = [None] * num_samples   # images not used in demo mode

    # Ground-truth distances in meters (synthetic but realistic)
    true_distances = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    # ---- Run calibration ----
    s = run_calibration_pipeline(
        images=fake_images,
        true_distances=true_distances,
        detection_mode="demo",
        depth_mode="demo",
    )

    print("Estimated scale factor s:", s)
