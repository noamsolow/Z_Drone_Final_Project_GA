from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import numpy as np
from PIL import Image


def fake_depth_model(image):
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.float32)
    return np.asarray(image, dtype=np.float32)


def make_yolo_line(image_width, image_height, x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    width_norm = width / float(image_width)
    height_norm = height / float(image_height)
    return "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
        x_center,
        y_center,
        width_norm,
        height_norm,
    )


def create_sample_dataset(root: Path, repeats_per_stratum: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)
    image_width = 24
    image_height = 24
    stratum_specs = [
        ("near", 30, "clear_sky", "10AM", (8, 8)),
        ("near", 40, "clear_sky", "8PM", (8, 7)),
        ("near", 50, "light_rain", "10AM", (7, 8)),
        ("near", 60, "light_rain", "8PM", (7, 7)),
        ("mid", 70, "clear_sky", "10AM", (6, 6)),
        ("mid", 80, "clear_sky", "8PM", (6, 5)),
        ("mid", 90, "light_rain", "10AM", (5, 6)),
        ("mid", 100, "light_rain", "8PM", (5, 5)),
        ("far", 115, "clear_sky", "10AM", (4, 4)),
        ("far", 125, "clear_sky", "8PM", (4, 3)),
        ("far", 125, "light_rain", "10AM", (3, 4)),
        ("far", 150, "light_rain", "8PM", (3, 3)),
    ]

    sample_index = 1
    for _, distance, weather, time_of_day, (bbox_width, bbox_height) in stratum_specs:
        for repeat_index in range(repeats_per_stratum):
            file_stem = "HighresScreenshot{:05d}_depth_{}_{}_{}".format(
                sample_index,
                distance,
                weather,
                time_of_day,
            )
            image_path = root / "{}.png".format(file_stem)
            label_path = root / "{}.txt".format(file_stem)

            pixel_value = min(255, int(distance + repeat_index))
            image = Image.new("RGB", (image_width, image_height), color=(pixel_value,) * 3)
            image.save(image_path)

            x1 = 6
            y1 = 6
            x2 = x1 + bbox_width
            y2 = y1 + bbox_height
            label_path.write_text(
                make_yolo_line(image_width, image_height, x1, y1, x2, y2),
                encoding="utf-8",
            )
            sample_index += 1


def create_temp_config(dataset_root: Path, output_root: Path) -> Dict[str, object]:
    return {
        "dataset_root": str(dataset_root),
        "strict": True,
        "max_samples": None,
        "depth_model": {
            "model_name": "unused-in-tests",
            "device": None,
        },
        "split": {
            "seed": 123,
            "holdout_fraction": 0.15,
            "num_folds": 5,
        },
        "output_root": str(output_root),
    }
