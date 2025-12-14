"""
Drone detection module.

Contains:
1. A real YOLO-based drone detector.
2. A demo detector that returns realistic random detections.

Both implementations share the same interface and output format.
"""

import random


# =========================
# REAL DETECTOR (YOLO)
# =========================

def detect_drone(image, model, conf_threshold=0.25):
    """
    Detect a drone in an RGB image using a pretrained YOLO model.

    Parameters
    ----------
    image
        Input RGB image (numpy array or compatible format).
    model
        Loaded YOLO model (e.g., ultralytics.YOLO instance).
    conf_threshold : float
        Confidence threshold for detections.

    Returns
    -------
    dict
        Detection result:
        {
            "bbox": (x1, y1, x2, y2),
            "x_pixel": float,
            "y_pixel": float
        }

    Raises
    ------
    RuntimeError
        If no drone is detected.
    """

    # Run inference
    results = model(image, conf=conf_threshold)

    # Expecting a single image
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        raise RuntimeError("No drone detected in image.")

    # Select the highest-confidence detection
    best_box = None
    best_conf = -1.0

    for box in detections:
        conf = float(box.conf)
        if conf > best_conf:
            best_conf = conf
            best_box = box

    # Extract bounding box (xyxy format)
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    return {
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
        "x_pixel": x_center,
        "y_pixel": y_center,
    }


# =========================
# DEMO DETECTOR
# =========================

def detect_drone_demo(image, image_width=640, image_height=480):
    """
    Demo implementation of drone detection.

    Returns a random but realistic bounding box to simulate
    drone detections without running a heavy model.

    Parameters
    ----------
    image
        Input RGB image (unused).
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    dict
        Detection result:
        {
            "bbox": (x1, y1, x2, y2),
            "x_pixel": float,
            "y_pixel": float
        }
    """

    # Drone size relative to image (typical far-field drone)
    box_width = random.uniform(0.04, 0.12) * image_width
    box_height = random.uniform(0.03, 0.10) * image_height

    # Drone usually appears in upper/middle part of frame
    center_x = random.uniform(0.25, 0.75) * image_width
    center_y = random.uniform(0.20, 0.60) * image_height

    x1 = int(center_x - box_width / 2)
    y1 = int(center_y - box_height / 2)
    x2 = int(center_x + box_width / 2)
    y2 = int(center_y + box_height / 2)

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width - 1, x2)
    y2 = min(image_height - 1, y2)

    return {
        "bbox": (x1, y1, x2, y2),
        "x_pixel": center_x,
        "y_pixel": center_y,
    }
