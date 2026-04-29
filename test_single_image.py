"""
Visual integration test for one RANDOM drone localization sample.

This script verifies the current Depth -> Mask path:

1. Load a random dataset sample and its single YOLO bbox.
2. Run Depth Anything V2 on the full original image.
3. Extract the median relative depth inside the bbox.
4. Plot the RGB image and depth map side by side with the same bbox overlay.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pipeline.data.loader import DEFAULT_DATASET_ROOT, DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.drone_depth import extract_drone_relative_depth


def _draw_bbox(axis, bbox, edgecolor="red", linewidth=2.0):
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


def main():
    dataset_root = Path(DEFAULT_DATASET_ROOT)

    print("Searching for samples in dataset:")
    print("  {}".format(dataset_root))

    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=False)
    
    # 1. המרת ה-Generator לרשימה ובחירת דגימה אקראית
    samples = list(loader.iter_samples())
    if not samples:
        raise RuntimeError("No valid .png samples with matching YOLO labels were found.")
    
    sample = random.choice(samples)
    print("Selected random sample: {}".format(sample.image_path.name))

    image = loader.load_image(sample)
    bbox = sample.annotation.bbox

    print("Loading Depth Anything V2 model...")
    depth_model = load_depth_anything_v2()

    print("Running full-image depth inference...")
    depth_map = estimate_relative_depth(image, depth_model)
    median_relative_depth = extract_drone_relative_depth(depth_map, bbox)

    print("\nSingle Image Integration Test (Random)")
    print("-----------------------------")
    print("Image: {}".format(sample.image_path))
    print("Label: {}".format(sample.label_path))
    print("True Distance (Z_true): {:.2f} m".format(sample.true_distance_m))
    print("Extracted Median Relative Depth: {:.6f}".format(median_relative_depth))
    print("Weather: {}".format(sample.weather))
    print("Time: {}".format(sample.time_of_day))
    print("BBox (x1, y1, x2, y2): {}".format(bbox))
    print("Image size (W x H): {} x {}".format(sample.image_width, sample.image_height))
    print("Depth map shape (H x W): {}".format(depth_map.shape))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    axes[0].imshow(image)
    _draw_bbox(axes[0], bbox)
    axes[0].set_title("Original RGB | Z_true = {:.2f} m".format(sample.true_distance_m))
    axes[0].axis("off")

    depth_plot = axes[1].imshow(depth_map, cmap="inferno")
    _draw_bbox(axes[1], bbox)
    axes[1].set_title(
        "Depth Map | Median Relative Depth = {:.6f}".format(median_relative_depth)
    )
    axes[1].axis("off")

    fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04, label="Relative depth")
    
    # 2. שמירה לקובץ במקום plt.show() כדי לעקוף את בעיית ה-Non-interactive backend בשרת
    plt.savefig('test_output.png', bbox_inches='tight')
    print("\nVisual test results saved to test_output.png!")


if __name__ == "__main__":
    main()