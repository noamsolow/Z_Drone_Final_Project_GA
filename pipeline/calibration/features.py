"""
Feature extraction for the first calibration study.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pipeline.data import DEFAULT_DATASET_ROOT, DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth
from pipeline.depth.drone_depth import extract_drone_relative_depth


CONTINUOUS_FEATURE_NAMES = [
    "relative_depth",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
]


@dataclass(frozen=True)
class RegressionRecord:
    """Flat per-sample feature row for the calibration study."""

    image_path: str
    label_path: str
    true_distance_m: float
    relative_depth: float
    bbox_width_px: int
    bbox_height_px: int
    bbox_width_norm: float
    bbox_height_norm: float
    bbox_area_ratio: float
    bbox_aspect_ratio: float
    bbox_center_x_norm: float
    bbox_center_y_norm: float
    weather: str
    time_of_day: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RegressionRecord":
        parsed = dict(values)
        parsed["true_distance_m"] = float(parsed["true_distance_m"])
        parsed["relative_depth"] = float(parsed["relative_depth"])
        parsed["bbox_width_px"] = int(parsed["bbox_width_px"])
        parsed["bbox_height_px"] = int(parsed["bbox_height_px"])
        parsed["bbox_width_norm"] = float(parsed["bbox_width_norm"])
        parsed["bbox_height_norm"] = float(parsed["bbox_height_norm"])
        parsed["bbox_area_ratio"] = float(parsed["bbox_area_ratio"])
        parsed["bbox_aspect_ratio"] = float(parsed["bbox_aspect_ratio"])
        parsed["bbox_center_x_norm"] = float(parsed["bbox_center_x_norm"])
        parsed["bbox_center_y_norm"] = float(parsed["bbox_center_y_norm"])
        parsed["weather"] = str(parsed["weather"])
        parsed["time_of_day"] = str(parsed["time_of_day"])
        parsed["image_path"] = str(parsed["image_path"])
        parsed["label_path"] = str(parsed["label_path"])
        return cls(**parsed)


REGRESSION_RECORD_FIELDNAMES = [field.name for field in fields(RegressionRecord)]


def _round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _build_record(sample, relative_depth: float) -> RegressionRecord:
    x1, y1, x2, y2 = sample.annotation.bbox
    bbox_width_px = int(x2 - x1)
    bbox_height_px = int(y2 - y1)

    if bbox_width_px <= 0 or bbox_height_px <= 0:
        raise ValueError("Sample has a non-positive bounding box.")

    image_width = int(sample.image_width)
    image_height = int(sample.image_height)
    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0

    return RegressionRecord(
        image_path=str(sample.image_path),
        label_path=str(sample.label_path),
        true_distance_m=_round_float(sample.true_distance_m),
        relative_depth=_round_float(relative_depth),
        bbox_width_px=bbox_width_px,
        bbox_height_px=bbox_height_px,
        bbox_width_norm=_round_float(bbox_width_px / float(image_width)),
        bbox_height_norm=_round_float(bbox_height_px / float(image_height)),
        bbox_area_ratio=_round_float(
            (bbox_width_px * bbox_height_px) / float(image_width * image_height)
        ),
        bbox_aspect_ratio=_round_float(bbox_width_px / float(bbox_height_px)),
        bbox_center_x_norm=_round_float(bbox_center_x / float(image_width)),
        bbox_center_y_norm=_round_float(bbox_center_y / float(image_height)),
        weather=str(sample.weather),
        time_of_day=str(sample.time_of_day),
    )


def build_regression_dataset(
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    depth_model: Any = None,
    strict: bool = True,
    max_samples: Optional[int] = None,
) -> List[RegressionRecord]:
    """
    Build flat regression-ready records from the dataset loader and depth model.
    """
    if depth_model is None:
        raise ValueError("depth_model must be provided to build the regression dataset.")

    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=strict)
    records: List[RegressionRecord] = []

    for sample_index, sample in enumerate(loader.iter_samples()):
        if max_samples is not None and sample_index >= max_samples:
            break

        image = loader.load_image(sample)
        depth_map = estimate_relative_depth(image, depth_model)
        relative_depth = extract_drone_relative_depth(depth_map, sample.annotation.bbox)
        records.append(_build_record(sample, relative_depth))

    return records


def write_regression_records_csv(
    records: Sequence[RegressionRecord],
    output_path: Union[str, Path],
) -> Path:
    """Write regression records to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGRESSION_RECORD_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

    return path


def read_regression_records_csv(input_path: Union[str, Path]) -> List[RegressionRecord]:
    """Read regression records from CSV."""
    path = Path(input_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [RegressionRecord.from_dict(row) for row in reader]


def summarize_regression_records(records: Iterable[RegressionRecord]) -> Dict[str, Any]:
    """Create a compact summary suitable for JSON output."""
    rows = list(records)
    weather_counts = Counter(record.weather for record in rows)
    time_counts = Counter(record.time_of_day for record in rows)
    distances = sorted({record.true_distance_m for record in rows})

    return {
        "num_records": len(rows),
        "weather_counts": dict(weather_counts),
        "time_of_day_counts": dict(time_counts),
        "unique_true_distances_m": distances,
    }
