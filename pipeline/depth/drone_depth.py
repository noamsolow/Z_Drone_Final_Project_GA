"""
Relative depth extraction at the detected drone location.

Implements the step described in the paper as:
'relative depth prediction corresponding to the detected drone'.
"""

import numpy as np


def extract_drone_relative_depth(depth_map, bbox):
    """
    Extract the relative depth prediction corresponding to the detected drone.

    Parameters
    ----------
    depth_map
        Relative depth map produced by the monocular depth model (H x W).
    bbox
        Bounding box of the detected drone: (x1, y1, x2, y2).

    Returns
    -------
    float
        Relative depth value corresponding to the detected drone.
    """
    x1, y1, x2, y2 = bbox

    h, w = depth_map.shape[:2]

    # Clamp bounding box to valid image coordinates
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid drone bounding box.")

    # Depth values corresponding to the detected drone
    drone_depth_values = depth_map[y1:y2, x1:x2]

    if drone_depth_values.size == 0:
        raise ValueError("No depth values found at drone location.")

    # Representative relative depth (robust to noise)
    relative_depth = float(np.median(drone_depth_values))

    return relative_depth
