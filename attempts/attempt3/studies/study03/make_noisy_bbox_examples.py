"""
Create visual examples for Study 03 noisy-bbox aggregation.

Each output figure shows:
- original RGB image with the exact bbox
- original RGB image with the jittered boxes
- one context crop with a depth map and the exact local bbox
- the same depth map with all jittered local boxes

If a predictions CSV is supplied, the title also includes the model prediction
for that image.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study03.run_study_03 import (
    DEFAULT_CONFIG_PATH,
    _jitter_bbox,
    _normalize_path,
    _resolve_context_and_aggregation_lists,
    _resolve_jitter_specs,
)
from pipeline.data.loader import DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import clamp_bbox, extract_context_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create noisy-bbox visualization examples for Study 03.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 03 YAML config.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="Number of examples to render.",
    )
    parser.add_argument(
        "--context-name",
        type=str,
        default="bbox_expand_2x",
        help="Which context crop to visualize in the depth-map panels.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional predictions CSV to annotate metric predictions.",
    )
    parser.add_argument(
        "--prediction-model-name",
        type=str,
        default=None,
        help="Optional model_name filter when reading predictions CSV.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Optional split_name filter when reading predictions CSV.",
    )
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_config(config_path: Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_predictions(
    input_path: Optional[Path],
    prediction_model_name: Optional[str],
    split_name: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    if input_path is None:
        return {}
    resolved_path = _resolve_repo_path(str(input_path))
    if not resolved_path.exists():
        return {}
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if prediction_model_name and str(row.get("model_name")) != str(prediction_model_name):
            continue
        if split_name and str(row.get("split_name")) != str(split_name):
            continue
        output[_normalize_path(str(row["image_path"]))] = dict(row)
    return output


def _draw_bbox(ax: Any, bbox: Sequence[int], color: str, label: Optional[str] = None, linewidth: float = 2.0) -> None:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )


def _pick_samples(samples: Sequence[Any], num_examples: int) -> List[Any]:
    sorted_samples = sorted(
        samples,
        key=lambda sample: (
            float(sample.true_distance_m),
            str(sample.weather),
            str(sample.time_of_day),
            str(sample.image_path),
        ),
    )
    if num_examples >= len(sorted_samples):
        return list(sorted_samples)
    indices = np.linspace(0, len(sorted_samples) - 1, num_examples, dtype=int)
    return [sorted_samples[int(index)] for index in indices]


def _resolve_context_spec(study_config: Dict[str, Any], context_name: str) -> Any:
    context_specs, _ = _resolve_context_and_aggregation_lists(study_config)
    matches = [spec for spec in context_specs if str(spec.name) == str(context_name)]
    if not matches:
        raise ValueError("Context '{}' not found in config.".format(context_name))
    return matches[0]


def _prediction_text(prediction_row: Optional[Dict[str, Any]]) -> str:
    if not prediction_row:
        return "pred: n/a"
    if "predicted_distance_m" in prediction_row:
        return "pred: {:.2f}m".format(float(prediction_row["predicted_distance_m"]))
    if "prediction" in prediction_row:
        return "pred: {:.2f}m".format(float(prediction_row["prediction"]))
    return "pred: n/a"


def _save_example_figure(
    output_path: Path,
    sample: Any,
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    exact_bbox: Sequence[int],
    jittered_bboxes: Sequence[Sequence[int]],
    exact_local_bbox: Sequence[int],
    jittered_local_bboxes: Sequence[Sequence[int]],
    prediction_row: Optional[Dict[str, Any]],
    context_name: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original RGB + exact bbox")
    _draw_bbox(axes[0, 0], exact_bbox, color="#00ff66", label="exact", linewidth=2.4)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_rgb)
    axes[0, 1].set_title("Original RGB + jitter boxes")
    _draw_bbox(axes[0, 1], exact_bbox, color="#00ff66", label="exact", linewidth=2.0)
    palette = ["#ff6b6b", "#ffd166", "#4cc9f0", "#f72585", "#90be6d", "#f8961e", "#577590"]
    for jitter_index, jitter_bbox in enumerate(jittered_bboxes):
        color = palette[jitter_index % len(palette)]
        _draw_bbox(axes[0, 1], jitter_bbox, color=color, label=str(jitter_index + 1), linewidth=1.6)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(depth_map, cmap="inferno")
    axes[1, 0].set_title("Depth map ({}) + exact local bbox".format(context_name))
    _draw_bbox(axes[1, 0], exact_local_bbox, color="#00ff66", label="exact", linewidth=2.4)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(depth_map, cmap="inferno")
    axes[1, 1].set_title("Depth map ({}) + jittered local boxes".format(context_name))
    _draw_bbox(axes[1, 1], exact_local_bbox, color="#00ff66", label="exact", linewidth=2.0)
    for jitter_index, jitter_bbox in enumerate(jittered_local_bboxes):
        color = palette[jitter_index % len(palette)]
        _draw_bbox(axes[1, 1], jitter_bbox, color=color, label=str(jitter_index + 1), linewidth=1.6)
    axes[1, 1].axis("off")

    title = (
        "{} | true: {:.1f}m | {} | weather={} | time={}".format(
            Path(sample.image_path).name,
            float(sample.true_distance_m),
            _prediction_text(prediction_row),
            str(sample.weather),
            str(sample.time_of_day),
        )
    )
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    resolved_config = _load_config(args.config)
    dataset_root = _resolve_repo_path(str(resolved_config["dataset_root"]))
    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=bool(resolved_config.get("strict", True)))
    all_samples = list(loader.iter_samples())
    chosen_samples = _pick_samples(all_samples, max(1, int(args.num_examples)))

    study_config = dict(resolved_config.get("study", {}))
    jitter_specs = _resolve_jitter_specs(study_config)
    context_spec = _resolve_context_spec(study_config, args.context_name)

    depth_config = dict(resolved_config.get("depth_model", {}))
    depth_model = load_depth_anything_v2(
        model_name=str(depth_config.get("model_name")) if depth_config.get("model_name") else None,
        device=depth_config.get("device"),
        local_files_only=bool(depth_config.get("local_files_only", False)),
    )

    predictions_by_image = _load_predictions(
        input_path=args.predictions_csv,
        prediction_model_name=args.prediction_model_name,
        split_name=args.split_name,
    )

    output_dir = _resolve_repo_path(str(resolved_config["output_root"])) / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, sample in enumerate(chosen_samples, start=1):
        image = loader.load_image(sample)
        image_rgb = np.asarray(image)
        exact_bbox = sample.annotation.bbox
        jittered_bboxes = [
            _jitter_bbox(
                bbox=exact_bbox,
                image_width=int(sample.image_width),
                image_height=int(sample.image_height),
                jitter_spec=jitter_spec,
            )
            for jitter_spec in jitter_specs
        ]

        crop_data = extract_context_crop(
            image=image,
            bbox=exact_bbox,
            context_spec=context_spec,
        )
        depth_map = estimate_relative_depth(crop_data["image"], depth_model)
        exact_local_bbox = crop_data["local_bbox"]
        jittered_local_bboxes = [
            (
                jitter_bbox[0] - crop_data["crop_bbox"][0],
                jitter_bbox[1] - crop_data["crop_bbox"][1],
                jitter_bbox[2] - crop_data["crop_bbox"][0],
                jitter_bbox[3] - crop_data["crop_bbox"][1],
            )
            for jitter_bbox in jittered_bboxes
        ]
        crop_width = int(depth_map.shape[1])
        crop_height = int(depth_map.shape[0])
        jittered_local_bboxes = [
            clamp_bbox(
                bbox=bbox,
                image_width=crop_width,
                image_height=crop_height,
            )
            for bbox in jittered_local_bboxes
        ]

        prediction_row = predictions_by_image.get(_normalize_path(str(sample.image_path)))
        output_path = output_dir / "example_{:02d}_{}.png".format(sample_index, Path(sample.image_path).stem)
        _save_example_figure(
            output_path=output_path,
            sample=sample,
            image_rgb=image_rgb,
            depth_map=depth_map,
            exact_bbox=exact_bbox,
            jittered_bboxes=jittered_bboxes,
            exact_local_bbox=exact_local_bbox,
            jittered_local_bboxes=jittered_local_bboxes,
            prediction_row=prediction_row,
            context_name=str(context_spec.name),
        )
        print("[study03-examples] wrote {}".format(output_path))


if __name__ == "__main__":
    main()
