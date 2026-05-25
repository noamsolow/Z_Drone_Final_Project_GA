"""
Visual integration test for one user-selected drone image.

This script verifies the current Depth -> Mask path:

1. Ask for an existing image path, unless one is passed on the command line.
2. Load the matching YOLO bbox from the sibling ``.txt`` file.
3. Run Depth Anything V2 on the full original image.
4. Extract the median relative depth inside the bbox.
5. Save three panels as image files in the project folder:
   - original RGB image
   - depth map
   - depth map with the drone bbox overlay
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.loader import load_image_rgb, parse_yolo_label_file
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2


def _draw_bbox(axis, bbox, edgecolor="cyan", linewidth=2.0):
    """Draw an ``(x1, y1, x2, y2)`` bbox on a matplotlib axis."""
    x1, y1, x2, y2 = bbox
    rectangle = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
    )
    axis.add_patch(rectangle)


def _clamp_bbox(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box after clamping: {}".format(bbox))

    return x1, y1, x2, y2


def _resize_bbox_about_center(bbox, scale, image_width, image_height):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = (x2 - x1) * float(scale)
    height = (y2 - y1) * float(scale)

    resized_bbox = (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )
    return _clamp_bbox(resized_bbox, image_width, image_height)


def _extract_bbox_values(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    values = depth_map[y1:y2, x1:x2]
    if values.size == 0:
        raise ValueError("No depth values found inside bbox: {}".format(bbox))
    return values


def _extract_ring_values(depth_map, inner_bbox, outer_bbox):
    outer_x1, outer_y1, outer_x2, outer_y2 = outer_bbox
    inner_x1, inner_y1, inner_x2, inner_y2 = inner_bbox

    outer_patch = depth_map[outer_y1:outer_y2, outer_x1:outer_x2]
    if outer_patch.size == 0:
        raise ValueError("No depth values found inside outer bbox: {}".format(outer_bbox))

    ring_mask = np.ones(outer_patch.shape, dtype=bool)
    ring_mask[
        inner_y1 - outer_y1 : inner_y2 - outer_y1,
        inner_x1 - outer_x1 : inner_x2 - outer_x1,
    ] = False

    ring_values = outer_patch[ring_mask]
    if ring_values.size == 0:
        raise ValueError("No surrounding ring values found for bbox: {}".format(inner_bbox))

    return ring_values


def _compute_depth_metrics(depth_map, bbox):
    image_height, image_width = depth_map.shape[:2]
    full_bbox = _clamp_bbox(bbox, image_width, image_height)

    # Focus on the center of the detection so background pixels near the box
    # borders do not dominate the representative drone depth estimate.
    inner_bbox = _resize_bbox_about_center(
        full_bbox,
        scale=0.5,
        image_width=image_width,
        image_height=image_height,
    )
    surrounding_bbox = _resize_bbox_about_center(
        full_bbox,
        scale=1.8,
        image_width=image_width,
        image_height=image_height,
    )

    full_bbox_values = _extract_bbox_values(depth_map, full_bbox)
    inner_bbox_values = _extract_bbox_values(depth_map, inner_bbox)
    ring_values = _extract_ring_values(depth_map, full_bbox, surrounding_bbox)

    full_bbox_median = float(np.median(full_bbox_values))
    object_depth = float(np.median(inner_bbox_values))
    surrounding_depth = float(np.median(ring_values))
    depth_contrast = abs(object_depth - surrounding_depth)

    robust_global_range = float(np.percentile(depth_map, 95) - np.percentile(depth_map, 5))
    if robust_global_range <= 1e-6:
        confidence_score = 0.0
    else:
        normalized_contrast = depth_contrast / robust_global_range
        confidence_score = float(np.clip(normalized_contrast, 0.0, 1.0))

    if confidence_score < 0.08:
        confidence_label = "low"
    elif confidence_score < 0.18:
        confidence_label = "medium"
    else:
        confidence_label = "high"

    return {
        "full_bbox": full_bbox,
        "inner_bbox": inner_bbox,
        "surrounding_bbox": surrounding_bbox,
        "full_bbox_median": full_bbox_median,
        "object_depth": object_depth,
        "surrounding_depth": surrounding_depth,
        "depth_contrast": float(depth_contrast),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the depth model on one existing image and visualize its bbox."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Optional path to the image. If omitted, the script will prompt for it.",
    )
    return parser.parse_args()


def _resolve_image_path(image_path_argument=None):
    raw_path = image_path_argument
    if raw_path is None:
        raw_path = input("Enter the full path to an existing image: ").strip()

    cleaned_path = str(raw_path).strip().strip('"')
    if not cleaned_path:
        raise ValueError("An image path is required.")

    image_path = Path(cleaned_path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError("Image file does not exist: {}".format(image_path))
    if not image_path.is_file():
        raise FileNotFoundError("Image path is not a file: {}".format(image_path))

    return image_path


def main():
    args = _parse_args()
    image_path = _resolve_image_path(args.image_path)
    label_path = image_path.with_suffix(".txt")

    if not label_path.exists():
        raise FileNotFoundError(
            "Matching YOLO label file was not found next to the image: {}".format(label_path)
        )

    print("Loading image: {}".format(image_path))
    image = load_image_rgb(image_path)
    image_width, image_height = image.size

    print("Loading bounding box from: {}".format(label_path))
    annotation = parse_yolo_label_file(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
    )
    bbox = annotation.bbox

    print("Loading Depth Anything V2 model...")
    depth_model = load_depth_anything_v2()

    print("Running full-image depth inference...")
    depth_map = estimate_relative_depth(image, depth_model)
    depth_metrics = _compute_depth_metrics(depth_map, bbox)
    drone_relative_depth = depth_metrics["object_depth"]

    print("\nSingle Image Integration Test")
    print("-----------------------------")
    print("Image: {}".format(image_path))
    print("Label: {}".format(label_path))
    print("BBox (x1, y1, x2, y2): {}".format(bbox))
    print("Inner bbox used for depth extraction: {}".format(depth_metrics["inner_bbox"]))
    print("Image size (W x H): {} x {}".format(image_width, image_height))
    print("Depth map shape (H x W): {}".format(depth_map.shape))
    print("Full-bbox median depth: {:.6f}".format(depth_metrics["full_bbox_median"]))
    print("Center-focused drone depth: {:.6f}".format(drone_relative_depth))
    print("Surrounding-ring median depth: {:.6f}".format(depth_metrics["surrounding_depth"]))
    print("Depth contrast vs. surrounding ring: {:.6f}".format(depth_metrics["depth_contrast"]))
    print(
        "Depth reliability score: {:.3f} ({})".format(
            depth_metrics["confidence_score"],
            depth_metrics["confidence_label"],
        )
    )

    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    depth_plot = axes[1].imshow(depth_map, cmap="inferno", vmin=depth_min, vmax=depth_max)
    axes[1].set_title("Depth Map")
    axes[1].axis("off")

    axes[2].imshow(depth_map, cmap="inferno", vmin=depth_min, vmax=depth_max)
    _draw_bbox(axes[2], bbox)
    _draw_bbox(axes[2], depth_metrics["inner_bbox"], edgecolor="lime", linewidth=1.8)
    axes[2].set_title(
        "Depth Map + Drone BB | Depth = {:.6f} | Confidence = {:.3f}".format(
            drone_relative_depth,
            depth_metrics["confidence_score"],
        )
    )
    axes[2].axis("off")

    fig.colorbar(
        depth_plot,
        ax=axes[1:],
        fraction=0.030,
        pad=0.02,
        label="Relative depth",
    )

    project_output_dir = Path(__file__).resolve().parent
    output_prefix = "{}".format(image_path.stem)
    combined_output_path = project_output_dir / "{}_depth_visualization.png".format(output_prefix)
    original_output_path = project_output_dir / "{}_original.png".format(output_prefix)
    depth_output_path = project_output_dir / "{}_depth_map.png".format(output_prefix)
    depth_bbox_output_path = project_output_dir / "{}_depth_map_with_bbox.png".format(output_prefix)

    plt.savefig(combined_output_path, bbox_inches="tight")
    print("\nCombined visualization saved to {}".format(combined_output_path))

    fig_original, axis_original = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis_original.imshow(image)
    axis_original.set_title("Original Image")
    axis_original.axis("off")
    fig_original.savefig(original_output_path, bbox_inches="tight")
    plt.close(fig_original)

    fig_depth, axis_depth = plt.subplots(figsize=(7, 6), constrained_layout=True)
    depth_plot_single = axis_depth.imshow(depth_map, cmap="inferno", vmin=depth_min, vmax=depth_max)
    axis_depth.set_title("Depth Map")
    axis_depth.axis("off")
    fig_depth.colorbar(depth_plot_single, ax=axis_depth, fraction=0.046, pad=0.04, label="Relative depth")
    fig_depth.savefig(depth_output_path, bbox_inches="tight")
    plt.close(fig_depth)

    fig_depth_bbox, axis_depth_bbox = plt.subplots(figsize=(7, 6), constrained_layout=True)
    depth_plot_bbox = axis_depth_bbox.imshow(
        depth_map,
        cmap="inferno",
        vmin=depth_min,
        vmax=depth_max,
    )
    _draw_bbox(axis_depth_bbox, bbox)
    _draw_bbox(axis_depth_bbox, depth_metrics["inner_bbox"], edgecolor="lime", linewidth=1.8)
    axis_depth_bbox.set_title(
        "Depth Map + Drone BB | Depth = {:.6f} | Confidence = {:.3f}".format(
            drone_relative_depth,
            depth_metrics["confidence_score"],
        )
    )
    axis_depth_bbox.axis("off")
    fig_depth_bbox.colorbar(
        depth_plot_bbox,
        ax=axis_depth_bbox,
        fraction=0.046,
        pad=0.04,
        label="Relative depth",
    )
    fig_depth_bbox.savefig(depth_bbox_output_path, bbox_inches="tight")
    plt.close(fig_depth_bbox)
    plt.close(fig)

    print("Original image saved to {}".format(original_output_path))
    print("Depth map saved to {}".format(depth_output_path))
    print("Depth map with bbox saved to {}".format(depth_bbox_output_path))


if __name__ == "__main__":
    main()
