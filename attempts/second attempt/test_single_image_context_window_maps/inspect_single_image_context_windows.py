"""
Generate relative-depth visualizations for one image across all study context windows.

This script is meant for visual inspection, not batch evaluation.

For one input image it will:

1. Load the sibling YOLO label by default.
2. Run the depth model on each context window used in the second-attempt study.
3. Save the regular image plus per-context crops and depth-map visualizations.
4. Print the important crop and depth-representation values to the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.loader import load_image_rgb, parse_yolo_label_file
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import (
    DEFAULT_CONTEXT_WINDOW_SPECS,
    compute_depth_representation,
    extract_context_crop,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one image across all second-attempt depth context windows."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the image. If omitted, the script will prompt for it.",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=None,
        help="Optional explicit label path. Defaults to the sibling .txt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to a per-image folder under this script folder.",
    )
    parser.add_argument(
        "--model-name",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Depth model name to load.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model downloads instead of using only the local Hugging Face cache.",
    )
    return parser.parse_args()


def _resolve_image_path(image_path_argument: Optional[str]) -> Path:
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


def _resolve_label_path(image_path: Path, explicit_label_path: Optional[Path]) -> Path:
    label_path = explicit_label_path or image_path.with_suffix(".txt")
    if not label_path.exists():
        raise FileNotFoundError(
            "Matching YOLO label file was not found: {}".format(label_path)
        )
    if not label_path.is_file():
        raise FileNotFoundError("Label path is not a file: {}".format(label_path))
    return label_path


def _draw_bbox(
    axis: Any,
    bbox: tuple[int, int, int, int],
    edgecolor: str = "cyan",
    linewidth: float = 2.0,
    label: str = "",
) -> None:
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
    if label:
        axis.text(
            x1,
            max(0, y1 - 3),
            label,
            color=edgecolor,
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5},
        )


def _save_original_views(image: Any, bbox: tuple[int, int, int, int], output_dir: Path, stem: str) -> None:
    original_path = output_dir / "{}_original.png".format(stem)
    image.save(original_path)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.imshow(image)
    _draw_bbox(axis, bbox, edgecolor="cyan", linewidth=2.0, label="drone bbox")
    axis.set_title("Original Image + Drone Bounding Box")
    axis.axis("off")
    figure.savefig(output_dir / "{}_original_with_bbox.png".format(stem), bbox_inches="tight")
    plt.close(figure)


def _save_context_outputs(
    context_name: str,
    crop_image: Any,
    local_bbox: tuple[int, int, int, int],
    focus_bbox: tuple[int, int, int, int],
    surrounding_bbox: tuple[int, int, int, int],
    depth_map: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    crop_image.save(output_dir / "{}_{}_crop_rgb.png".format(stem, context_name))

    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())

    figure_depth, axis_depth = plt.subplots(figsize=(8, 6), constrained_layout=True)
    depth_plot = axis_depth.imshow(depth_map, cmap="inferno", vmin=depth_min, vmax=depth_max)
    axis_depth.set_title("{} Depth Map".format(context_name))
    axis_depth.axis("off")
    figure_depth.colorbar(depth_plot, ax=axis_depth, fraction=0.046, pad=0.04, label="Relative depth")
    figure_depth.savefig(
        output_dir / "{}_{}_depth_map.png".format(stem, context_name),
        bbox_inches="tight",
    )
    plt.close(figure_depth)

    figure_overlay, axis_overlay = plt.subplots(figsize=(8, 6), constrained_layout=True)
    depth_plot_overlay = axis_overlay.imshow(depth_map, cmap="inferno", vmin=depth_min, vmax=depth_max)
    _draw_bbox(axis_overlay, local_bbox, edgecolor="cyan", linewidth=2.0, label="local bbox")
    _draw_bbox(axis_overlay, focus_bbox, edgecolor="lime", linewidth=1.8, label="focus bbox")
    _draw_bbox(
        axis_overlay,
        surrounding_bbox,
        edgecolor="magenta",
        linewidth=1.5,
        label="ring bbox",
    )
    axis_overlay.set_title("{} Depth Map + Analysis Boxes".format(context_name))
    axis_overlay.axis("off")
    figure_overlay.colorbar(
        depth_plot_overlay,
        ax=axis_overlay,
        fraction=0.046,
        pad=0.04,
        label="Relative depth",
    )
    figure_overlay.savefig(
        output_dir / "{}_{}_depth_map_with_boxes.png".format(stem, context_name),
        bbox_inches="tight",
    )
    plt.close(figure_overlay)


def _context_summary_row(
    context_name: str,
    crop_bbox: tuple[int, int, int, int],
    local_bbox: tuple[int, int, int, int],
    representation: Dict[str, Any],
    depth_map: np.ndarray,
) -> Dict[str, Any]:
    return {
        "context_window": context_name,
        "crop_bbox_global": list(crop_bbox),
        "local_bbox": list(local_bbox),
        "focus_bbox": list(representation["focus_bbox"]),
        "surrounding_bbox": list(representation["surrounding_bbox"]),
        "depth_map_shape": list(depth_map.shape),
        "depth_map_min": float(depth_map.min()),
        "depth_map_max": float(depth_map.max()),
        "depth_map_mean": float(np.mean(depth_map)),
        "depth_map_median": float(np.median(depth_map)),
        "object_depth": float(representation["object_depth"]),
        "object_depth_percentile_5_95": float(representation["object_depth_percentile_5_95"]),
        "surrounding_ring_depth": float(representation["surrounding_ring_depth"]),
        "object_minus_ring": float(representation["object_minus_ring"]),
        "object_minus_ring_normalized": float(representation["object_minus_ring_normalized"]),
        "global_percentile_5": float(representation["global_percentile_5"]),
        "global_percentile_95": float(representation["global_percentile_95"]),
        "robust_global_range": float(representation["robust_global_range"]),
    }


def _print_context_summary(row: Dict[str, Any]) -> None:
    print("\n[context] {}".format(row["context_window"]))
    print("  crop_bbox_global: {}".format(tuple(row["crop_bbox_global"])))
    print("  local_bbox: {}".format(tuple(row["local_bbox"])))
    print("  focus_bbox: {}".format(tuple(row["focus_bbox"])))
    print("  surrounding_bbox: {}".format(tuple(row["surrounding_bbox"])))
    print("  depth_map_shape: {}".format(tuple(row["depth_map_shape"])))
    print(
        "  depth_stats: min={:.6f}, max={:.6f}, mean={:.6f}, median={:.6f}".format(
            row["depth_map_min"],
            row["depth_map_max"],
            row["depth_map_mean"],
            row["depth_map_median"],
        )
    )
    print(
        "  representation: object_depth={:.6f}, percentile_5_95={:.6f}, ring_depth={:.6f}".format(
            row["object_depth"],
            row["object_depth_percentile_5_95"],
            row["surrounding_ring_depth"],
        )
    )
    print(
        "  contrast: object_minus_ring={:.6f}, normalized={:.6f}".format(
            row["object_minus_ring"],
            row["object_minus_ring_normalized"],
        )
    )
    print(
        "  global_range: p5={:.6f}, p95={:.6f}, robust_range={:.6f}".format(
            row["global_percentile_5"],
            row["global_percentile_95"],
            row["robust_global_range"],
        )
    )


def main() -> None:
    args = _parse_args()
    image_path = _resolve_image_path(args.image_path)
    label_path = _resolve_label_path(image_path, args.label_path)

    image = load_image_rgb(image_path)
    image_width, image_height = image.size
    annotation = parse_yolo_label_file(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
    )
    bbox = annotation.bbox

    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "outputs_{}".format(image_path.stem)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[single-image-contexts] Loading image: {}".format(image_path))
    print("[single-image-contexts] Loading label: {}".format(label_path))
    print("[single-image-contexts] Image size: {} x {}".format(image_width, image_height))
    print("[single-image-contexts] Global bbox: {}".format(bbox))
    print("[single-image-contexts] Output directory: {}".format(output_dir))
    print("[single-image-contexts] Loading depth model...")
    depth_model = load_depth_anything_v2(
        model_name=args.model_name,
        device=args.device,
        local_files_only=not args.allow_network,
    )

    _save_original_views(image, bbox, output_dir, image_path.stem)

    summary_rows = []
    for context_spec in DEFAULT_CONTEXT_WINDOW_SPECS:
        print("\n[single-image-contexts] Running context: {}".format(context_spec.name))
        crop_data = extract_context_crop(
            image=image,
            bbox=bbox,
            context_spec=context_spec,
        )
        depth_map = estimate_relative_depth(crop_data["image"], depth_model)
        representation = compute_depth_representation(
            depth_map=depth_map,
            bbox=crop_data["local_bbox"],
            aggregation_method="bbox_midpoint",
            inner_bbox_scale=0.5,
            surrounding_bbox_scale=1.8,
        )
        _save_context_outputs(
            context_name=context_spec.name,
            crop_image=crop_data["image"],
            local_bbox=crop_data["local_bbox"],
            focus_bbox=representation["focus_bbox"],
            surrounding_bbox=representation["surrounding_bbox"],
            depth_map=depth_map,
            output_dir=output_dir,
            stem=image_path.stem,
        )
        summary_row = _context_summary_row(
            context_name=context_spec.name,
            crop_bbox=crop_data["crop_bbox"],
            local_bbox=crop_data["local_bbox"],
            representation=representation,
            depth_map=depth_map,
        )
        summary_rows.append(summary_row)
        _print_context_summary(summary_row)

    summary_payload = {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_size": [image_width, image_height],
        "global_bbox": list(bbox),
        "contexts": summary_rows,
    }
    summary_path = output_dir / "{}_context_window_summary.json".format(image_path.stem)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n[single-image-contexts] Summary JSON saved to {}".format(summary_path))
    print("[single-image-contexts] Saved original image plus per-context RGB/depth outputs.")


if __name__ == "__main__":
    main()
