"""
Relative depth estimation using Depth Anything.
"""

import numpy as np


def estimate_relative_depth(image, depth_model):
    """
    Estimate a relative depth map from an RGB image
    using a pretrained Depth Anything model.

    Parameters
    ----------
    image
        Input RGB image.
    depth_model
        Loaded Depth Anything model.

    Returns
    -------
    numpy.ndarray
        Relative depth map (H x W), unitless.
    """
    depth_map = depth_model(image)

    # Depth Anything already returns a valid relative depth map
    depth_map = np.asarray(depth_map, dtype=np.float32)

    return depth_map


def estimate_relative_depth_demo(image, width=640, height=480):
    """
    Demo relative depth map for baseline testing.
    """
    import numpy as np

    depth = np.linspace(0.5, 2.5, height * width)
    return depth.reshape((height, width)).astype(np.float32)
