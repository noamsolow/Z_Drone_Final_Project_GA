"""
Utilities for studying how drone-relative depth should be represented.

The core idea is to separate two questions:

1. What image context should be sent into the monocular depth model?
2. Once a depth map is produced, how should the drone depth be summarized?

This module provides deterministic helpers for both parts so higher-level
experiment code can compare context windows and aggregation methods cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


SUPPORTED_AGGREGATION_METHODS = (
    "inner50_median",
    "bbox_mean",
    "bbox_midpoint",
)


@dataclass(frozen=True)
class ContextWindowSpec:
    """One depth-inference context window centered on the detected drone."""

    name: str
    scale: Optional[float]


DEFAULT_CONTEXT_WINDOW_SPECS = (
    ContextWindowSpec(name="full_image", scale=None),
    ContextWindowSpec(name="bbox_expand_4x", scale=4.0),
    ContextWindowSpec(name="bbox_expand_2x", scale=2.0),
    ContextWindowSpec(name="bbox_expand_1_5x", scale=1.5),
    ContextWindowSpec(name="bbox_only", scale=1.0),
)


def _to_pil_rgb_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Normalize supported image inputs into a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            return opened_image.convert("RGB").copy()

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim not in (2, 3):
            raise ValueError("Expected a 2D or 3D image array, got shape {}".format(array.shape))

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.size > 0 and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    raise TypeError("Unsupported image type: {}".format(type(image).__name__))


def clamp_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Clamp an ``(x1, y1, x2, y2)`` bbox to valid exclusive image bounds."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box after clamping: {}".format(tuple(bbox)))

    return x1, y1, x2, y2


def resize_bbox_about_center(
    bbox: Sequence[float],
    scale: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Resize a bbox about its center and clamp it to the image bounds."""
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = (x2 - x1) * float(scale)
    height = (y2 - y1) * float(scale)

    return clamp_bbox(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        image_width=image_width,
        image_height=image_height,
    )


def resolve_context_window_specs(
    raw_specs: Optional[Sequence[Union[ContextWindowSpec, Dict[str, Any]]]] = None,
) -> Tuple[ContextWindowSpec, ...]:
    """Resolve config dictionaries into immutable context-window specs."""
    if raw_specs is None:
        return DEFAULT_CONTEXT_WINDOW_SPECS

    resolved = []
    for raw_spec in raw_specs:
        if isinstance(raw_spec, ContextWindowSpec):
            resolved.append(raw_spec)
            continue

        if not isinstance(raw_spec, dict):
            raise TypeError("Context window spec must be a dict or ContextWindowSpec.")

        name = str(raw_spec["name"])
        scale = raw_spec.get("scale")
        scale = None if scale is None else float(scale)
        resolved.append(ContextWindowSpec(name=name, scale=scale))

    if not resolved:
        raise ValueError("At least one context window spec is required.")

    return tuple(resolved)


def extract_context_crop(
    image: Union[str, Path, Image.Image, np.ndarray],
    bbox: Sequence[int],
    context_spec: ContextWindowSpec,
) -> Dict[str, Any]:
    """
    Crop an image according to one context spec and translate the drone bbox.

    Returns a dictionary containing the cropped image, the crop bbox in global
    coordinates, and the drone bbox in crop-local coordinates.
    """
    pil_image = _to_pil_rgb_image(image)
    image_width, image_height = pil_image.size
    bbox = clamp_bbox(bbox, image_width=image_width, image_height=image_height)

    if context_spec.scale is None:
        crop_bbox = (0, 0, image_width, image_height)
    else:
        crop_bbox = resize_bbox_about_center(
            bbox=bbox,
            scale=context_spec.scale,
            image_width=image_width,
            image_height=image_height,
        )

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    local_bbox = (
        bbox[0] - crop_x1,
        bbox[1] - crop_y1,
        bbox[2] - crop_x1,
        bbox[3] - crop_y1,
    )
    local_bbox = clamp_bbox(
        local_bbox,
        image_width=crop_x2 - crop_x1,
        image_height=crop_y2 - crop_y1,
    )

    return {
        "context_name": context_spec.name,
        "crop_bbox": crop_bbox,
        "local_bbox": local_bbox,
        "image": pil_image.crop(crop_bbox),
    }


def _extract_bbox_values(depth_map: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width=depth_map.shape[1], image_height=depth_map.shape[0])
    values = depth_map[y1:y2, x1:x2]
    if values.size == 0:
        raise ValueError("No depth values found inside bbox: {}".format(tuple(bbox)))
    return values


def _extract_midpoint_depth(depth_map: np.ndarray, bbox: Sequence[int]) -> float:
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width=depth_map.shape[1], image_height=depth_map.shape[0])
    center_x = int((x1 + x2 - 1) / 2.0)
    center_y = int((y1 + y2 - 1) / 2.0)
    return float(depth_map[center_y, center_x])


def _extract_ring_values(
    depth_map: np.ndarray,
    inner_bbox: Sequence[int],
    outer_bbox: Sequence[int],
) -> np.ndarray:
    outer_x1, outer_y1, outer_x2, outer_y2 = clamp_bbox(
        outer_bbox,
        image_width=depth_map.shape[1],
        image_height=depth_map.shape[0],
    )
    inner_x1, inner_y1, inner_x2, inner_y2 = clamp_bbox(
        inner_bbox,
        image_width=depth_map.shape[1],
        image_height=depth_map.shape[0],
    )

    outer_patch = depth_map[outer_y1:outer_y2, outer_x1:outer_x2]
    if outer_patch.size == 0:
        raise ValueError("No depth values found inside outer bbox: {}".format(tuple(outer_bbox)))

    ring_mask = np.ones(outer_patch.shape, dtype=bool)
    ring_mask[
        inner_y1 - outer_y1 : inner_y2 - outer_y1,
        inner_x1 - outer_x1 : inner_x2 - outer_x1,
    ] = False

    ring_values = outer_patch[ring_mask]
    if ring_values.size == 0:
        raise ValueError("No surrounding ring values found for bbox: {}".format(tuple(inner_bbox)))

    return ring_values


def compute_depth_representation(
    depth_map: np.ndarray,
    bbox: Sequence[int],
    aggregation_method: str,
    inner_bbox_scale: float = 0.5,
    surrounding_bbox_scale: float = 1.8,
) -> Dict[str, Any]:
    """
    Compute one drone-depth representation and its normalized variants.

    The returned object depth depends on ``aggregation_method``:

    - ``inner50_median``: median inside the center-focused 50% bbox
    - ``bbox_mean``: mean over the full bbox
    - ``bbox_midpoint``: depth at the bbox midpoint pixel
    """
    if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
        raise ValueError(
            "Unsupported aggregation_method '{}'. Expected one of {}.".format(
                aggregation_method,
                SUPPORTED_AGGREGATION_METHODS,
            )
        )

    image_height, image_width = depth_map.shape[:2]
    full_bbox = clamp_bbox(bbox, image_width=image_width, image_height=image_height)
    try:
        focus_bbox = resize_bbox_about_center(
            bbox=full_bbox,
            scale=inner_bbox_scale,
            image_width=image_width,
            image_height=image_height,
        )
    except ValueError:
        # Very small or nearly-flat detections can collapse when shrunk around
        # center. In that case, fall back to the full bbox rather than failing
        # the entire study run.
        focus_bbox = full_bbox

    try:
        surrounding_bbox = resize_bbox_about_center(
            bbox=full_bbox,
            scale=surrounding_bbox_scale,
            image_width=image_width,
            image_height=image_height,
        )
    except ValueError:
        # Tight edge cases can also collapse when expanded and clamped near
        # image borders. Falling back to the full bbox keeps the representation
        # defined; the ring terms will naturally collapse to zero.
        surrounding_bbox = full_bbox

    if aggregation_method == "inner50_median":
        object_depth = float(np.median(_extract_bbox_values(depth_map, focus_bbox)))
    elif aggregation_method == "bbox_mean":
        object_depth = float(np.mean(_extract_bbox_values(depth_map, full_bbox)))
    else:
        object_depth = _extract_midpoint_depth(depth_map, full_bbox)

    try:
        ring_values = _extract_ring_values(depth_map, inner_bbox=full_bbox, outer_bbox=surrounding_bbox)
        surrounding_ring_depth = float(np.median(ring_values))
    except ValueError:
        # Tight crops such as ``bbox_only`` may leave no pixels outside the
        # detected drone box. In that case there is no usable local context, so
        # the context-relative terms collapse to zero by construction.
        surrounding_ring_depth = float(object_depth)

    percentile_5 = float(np.percentile(depth_map, 5))
    percentile_95 = float(np.percentile(depth_map, 95))
    robust_global_range = float(percentile_95 - percentile_5)

    if robust_global_range <= 1e-6:
        object_depth_percentile = 0.0
        object_minus_ring_normalized = 0.0
    else:
        object_depth_percentile = float(
            np.clip((object_depth - percentile_5) / robust_global_range, 0.0, 1.0)
        )
        object_minus_ring_normalized = float(
            (object_depth - surrounding_ring_depth) / robust_global_range
        )

    return {
        "full_bbox": full_bbox,
        "focus_bbox": focus_bbox,
        "surrounding_bbox": surrounding_bbox,
        "object_depth": object_depth,
        "surrounding_ring_depth": surrounding_ring_depth,
        "object_minus_ring": float(object_depth - surrounding_ring_depth),
        "object_depth_percentile_5_95": object_depth_percentile,
        "object_minus_ring_normalized": object_minus_ring_normalized,
        "robust_global_range": robust_global_range,
        "global_percentile_5": percentile_5,
        "global_percentile_95": percentile_95,
    }
