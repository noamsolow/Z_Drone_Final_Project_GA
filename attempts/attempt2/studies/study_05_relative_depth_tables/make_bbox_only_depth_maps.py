"""
Generate six bbox-only depth-map examples for Mini Study 05.

This utility selects:

- 2 images from 30m
- 2 images from 100m
- 2 images from 150m

For each selected image it:

1. Crops the RGB image to the exact drone bounding box (`bbox_only`).
2. Runs Depth Anything on that crop.
3. Saves a depth-map image and a side-by-side RGB/depth panel.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.loader import DEFAULT_DATASET_ROOT, DroneDatasetLoader
from pipeline.depth.depth_model import DEFAULT_DEPTH_ANYTHING_V2_MODEL
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import ContextWindowSpec, extract_context_crop


DEFAULT_DISTANCES = (30.0, 100.0, 150.0)
DEFAULT_SAMPLES_PER_DISTANCE = 2
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "bbox_only_depth_maps"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bbox-only depth-map examples for selected distances."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root containing the drone images.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        default=list(DEFAULT_DISTANCES),
        help="Exact ground-truth distances to sample from.",
    )
    parser.add_argument(
        "--samples-per-distance",
        type=int,
        default=DEFAULT_SAMPLES_PER_DISTANCE,
        help="How many images to save for each requested distance.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the example images will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_DEPTH_ANYTHING_V2_MODEL,
        help="Depth Anything model name or local model identifier.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional PyTorch device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the depth model from local cache only.",
    )
    return parser.parse_args()


def _sample_sort_key(sample) -> tuple:
    return (
        float(sample.true_distance_m),
        str(sample.weather),
        str(sample.time_of_day),
        str(sample.image_path),
    )


def _round_robin_diverse_pick(samples: Sequence, count: int) -> List:
    grouped: Dict[tuple, List] = defaultdict(list)
    for sample in sorted(samples, key=_sample_sort_key):
        grouped[(sample.weather, sample.time_of_day)].append(sample)

    selected: List = []
    keys = sorted(grouped.keys())
    index = 0

    while len(selected) < count:
        made_progress = False
        for key in keys:
            bucket = grouped[key]
            if index < len(bucket):
                selected.append(bucket[index])
                made_progress = True
                if len(selected) >= count:
                    break
        if not made_progress:
            break
        index += 1

    if len(selected) < count:
        raise ValueError(
            "Requested {} samples but only found {} usable samples.".format(
                count,
                len(selected),
            )
        )

    return selected[:count]


def _select_examples(dataset_root: Path, distances: Sequence[float], samples_per_distance: int) -> List:
    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=True)
    grouped: Dict[float, List] = defaultdict(list)

    for sample in loader.iter_samples():
        distance = float(sample.true_distance_m)
        if distance in distances:
            grouped[distance].append(sample)

    selected = []
    for distance in distances:
        distance_samples = grouped.get(float(distance), [])
        if len(distance_samples) < samples_per_distance:
            raise ValueError(
                "Distance {}m has only {} usable samples, but {} were requested.".format(
                    distance,
                    len(distance_samples),
                    samples_per_distance,
                )
            )
        selected.extend(_round_robin_diverse_pick(distance_samples, samples_per_distance))

    return selected


def _save_depth_map_figure(
    depth_map: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
    axis.imshow(depth_map, cmap="inferno")
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_panel_figure(
    crop_rgb,
    depth_map: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), constrained_layout=True)
    axes[0].imshow(crop_rgb)
    axes[0].set_title("BBox-only RGB crop", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(depth_map, cmap="inferno")
    axes[1].set_title("BBox-only depth map", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_contact_sheet(rendered_rows: Iterable[Dict[str, str]], output_path: Path) -> None:
    rows = list(rendered_rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for axis, row in zip(axes, rows):
        image = plt.imread(row["depth_map_png"])
        axis.imshow(image)
        axis.set_title(row["title"], fontsize=9)
        axis.axis("off")

    for axis in axes[len(rows) :]:
        axis.axis("off")

    fig.suptitle("Mini Study 05 | BBox-only depth maps", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slot_id",
        "true_distance_m",
        "weather",
        "time_of_day",
        "image_path",
        "label_path",
        "depth_map_png",
        "panel_png",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    distances = tuple(float(value) for value in args.distances)
    samples_per_distance = int(args.samples_per_distance)

    if samples_per_distance <= 0:
        raise ValueError("--samples-per-distance must be positive.")

    print("[study-05-depthmaps] Selecting samples")
    print("  dataset_root: {}".format(dataset_root))
    print("  distances: {}".format(", ".join(str(int(value)) for value in distances)))
    print("  samples_per_distance: {}".format(samples_per_distance))

    selected_samples = _select_examples(
        dataset_root=dataset_root,
        distances=distances,
        samples_per_distance=samples_per_distance,
    )

    print("[study-05-depthmaps] Loading depth model")
    print("  model_name: {}".format(args.model_name))
    print("  device: {}".format(args.device))
    print("  local_files_only: {}".format(bool(args.local_files_only)))
    depth_model = load_depth_anything_v2(
        model_name=args.model_name,
        device=args.device,
        local_files_only=bool(args.local_files_only),
    )

    bbox_only_spec = ContextWindowSpec(name="bbox_only", scale=1.0)
    rendered_rows: List[Dict[str, str]] = []

    total = len(selected_samples)
    for index, sample in enumerate(selected_samples, start=1):
        print(
            "[study-05-depthmaps] Rendering sample {}/{} | {} m | {} | {}".format(
                index,
                total,
                int(sample.true_distance_m),
                sample.weather,
                sample.time_of_day,
            )
        )

        image = DroneDatasetLoader.load_image(sample)
        crop_data = extract_context_crop(image, sample.annotation.bbox, bbox_only_spec)
        crop_rgb = crop_data["image"]
        depth_map = estimate_relative_depth(crop_rgb, depth_model)

        distance_slot = sum(
            1 for row in rendered_rows if float(row["true_distance_m"]) == float(sample.true_distance_m)
        ) + 1
        slot_id = "{}m_{:02d}".format(int(sample.true_distance_m), distance_slot)
        title = "{} | {} | {} | {}".format(
            slot_id,
            sample.weather,
            sample.time_of_day,
            sample.image_path.name,
        )

        depth_map_png = output_dir / "{}_depth_map.png".format(slot_id)
        panel_png = output_dir / "{}_panel.png".format(slot_id)

        _save_depth_map_figure(depth_map=depth_map, title=title, output_path=depth_map_png)
        _save_panel_figure(
            crop_rgb=crop_rgb,
            depth_map=depth_map,
            title=title,
            output_path=panel_png,
        )

        rendered_rows.append(
            {
                "slot_id": slot_id,
                "title": title,
                "true_distance_m": str(int(sample.true_distance_m)),
                "weather": str(sample.weather),
                "time_of_day": str(sample.time_of_day),
                "image_path": str(sample.image_path),
                "label_path": str(sample.label_path),
                "depth_map_png": str(depth_map_png),
                "panel_png": str(panel_png),
            }
        )

    summary_csv = output_dir / "selected_bbox_only_depth_maps.csv"
    contact_sheet_png = output_dir / "bbox_only_depth_maps_contact_sheet.png"
    _write_summary_csv(rendered_rows, summary_csv)
    _save_contact_sheet(rendered_rows, contact_sheet_png)

    print("[study-05-depthmaps] Complete")
    print("  output_dir: {}".format(output_dir))
    print("  summary_csv: {}".format(summary_csv))
    print("  contact_sheet_png: {}".format(contact_sheet_png))


if __name__ == "__main__":
    main()
