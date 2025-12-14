"""
3D position reconstruction using camera geometry.

Implements inverse projection from image coordinates
to camera-centric 3D coordinates, as described in the paper.
"""


def reconstruct_drone_position(x_pixel, y_pixel, z, intrinsics):
    """
    Reconstruct the 3D position of the drone in camera coordinates.

    Parameters
    ----------
    x_pixel
        X coordinate of the detected drone in image pixels.
    y_pixel
        Y coordinate of the detected drone in image pixels.
    z
        Estimated metric depth (Z_hat) in meters.
    intrinsics
        Camera intrinsic parameters:
        {
            "fx": float,
            "fy": float,
            "cx": float,
            "cy": float
        }

    Returns
    -------
    tuple
        (X, Y, Z) position of the drone in camera coordinates.
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    if fx == 0 or fy == 0:
        raise ValueError("Invalid camera intrinsics: fx and fy must be non-zero.")

    X = (x_pixel - cx) * z / fx
    Y = (y_pixel - cy) * z / fy
    Z = z

    return (X, Y, Z)
