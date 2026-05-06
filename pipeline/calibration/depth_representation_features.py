"""
Feature extraction for the second-attempt depth-representation study.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from pipeline.data import DEFAULT_DATASET_ROOT, DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth
from pipeline.depth.representation import (
    SUPPORTED_AGGREGATION_METHODS,
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEPTH_SCORE_FIELD_NAMES = [
    "object_depth",
    "object_depth_percentile_5_95",
    "object_minus_ring",
    "object_minus_ring_normalized",
]


@dataclass(frozen=True)
class DepthRepresentationRecord:
    """One sampled image evaluated with one context window and one aggregation."""

    image_path: str
    label_path: str
    stratum_key: str
    selection_rank_within_stratum: int
    true_distance_m: float
    weather: str
    time_of_day: str
    context_window: str
    context_scale: str
    aggregation_method: str
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    local_bbox_x1: int
    local_bbox_y1: int
    local_bbox_x2: int
    local_bbox_y2: int
    focus_bbox_x1: int
    focus_bbox_y1: int
    focus_bbox_x2: int
    focus_bbox_y2: int
    surrounding_bbox_x1: int
    surrounding_bbox_y1: int
    surrounding_bbox_x2: int
    surrounding_bbox_y2: int
    object_depth: float
    object_depth_percentile_5_95: float
    surrounding_ring_depth: float
    object_minus_ring: float
    object_minus_ring_normalized: float
    robust_global_range: float
    global_percentile_5: float
    global_percentile_95: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DepthRepresentationRecord":
        parsed = dict(values)
        int_fields = {
            "selection_rank_within_stratum",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "local_bbox_x1",
            "local_bbox_y1",
            "local_bbox_x2",
            "local_bbox_y2",
            "focus_bbox_x1",
            "focus_bbox_y1",
            "focus_bbox_x2",
            "focus_bbox_y2",
            "surrounding_bbox_x1",
            "surrounding_bbox_y1",
            "surrounding_bbox_x2",
            "surrounding_bbox_y2",
        }
        float_fields = {
            "true_distance_m",
            "object_depth",
            "object_depth_percentile_5_95",
            "surrounding_ring_depth",
            "object_minus_ring",
            "object_minus_ring_normalized",
            "robust_global_range",
            "global_percentile_5",
            "global_percentile_95",
        }
        for field_name in int_fields:
            parsed[field_name] = int(parsed[field_name])
        for field_name in float_fields:
            parsed[field_name] = float(parsed[field_name])
        for field_name in (
            "image_path",
            "label_path",
            "stratum_key",
            "weather",
            "time_of_day",
            "context_window",
            "context_scale",
            "aggregation_method",
        ):
            parsed[field_name] = str(parsed[field_name])
        return cls(**parsed)


DEPTH_REPRESENTATION_FIELDNAMES = [field.name for field in fields(DepthRepresentationRecord)]


@dataclass(frozen=True)
class SelectedSampleRecord:
    """One sampled dataset image chosen for the second-attempt mini-study."""

    image_path: str
    label_path: str
    stratum_key: str
    selection_rank_within_stratum: int
    true_distance_m: float
    weather: str
    time_of_day: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SELECTED_SAMPLE_FIELDNAMES = [field.name for field in fields(SelectedSampleRecord)]


def _round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _sample_stratum_key(sample: Any) -> Tuple[float, str, str]:
    return (float(sample.true_distance_m), str(sample.weather), str(sample.time_of_day))


def select_samples_per_stratum(
    samples: Sequence[Any],
    sample_count_per_stratum: int,
    seed: int = 0,
) -> List[Tuple[Any, int]]:
    """Select up to ``sample_count_per_stratum`` samples per metadata stratum."""
    if sample_count_per_stratum <= 0:
        raise ValueError("sample_count_per_stratum must be positive.")

    grouped: Dict[Tuple[float, str, str], List[Any]] = defaultdict(list)
    for sample in samples:
        grouped[_sample_stratum_key(sample)].append(sample)

    rng = np.random.default_rng(seed)
    selected: List[Tuple[Any, int]] = []
    for stratum_key in sorted(grouped):
        candidates = sorted(grouped[stratum_key], key=lambda item: str(item.image_path))
        candidate_indices = np.arange(len(candidates))
        rng.shuffle(candidate_indices)
        for selection_rank, candidate_index in enumerate(
            candidate_indices[:sample_count_per_stratum],
            start=1,
        ):
            selected.append((candidates[int(candidate_index)], selection_rank))

    selected.sort(
        key=lambda item: (
            item[0].true_distance_m,
            item[0].weather,
            item[0].time_of_day,
            str(item[0].image_path),
        )
    )
    return selected


def build_selected_sample_records(
    selected_samples: Sequence[Tuple[Any, int]],
) -> List[SelectedSampleRecord]:
    """Convert selected dataset samples into flat artifact rows."""
    rows: List[SelectedSampleRecord] = []
    for sample, selection_rank in selected_samples:
        rows.append(
            SelectedSampleRecord(
                image_path=str(sample.image_path),
                label_path=str(sample.label_path),
                stratum_key="{}|{}|{}".format(
                    _round_float(sample.true_distance_m),
                    sample.weather,
                    sample.time_of_day,
                ),
                selection_rank_within_stratum=int(selection_rank),
                true_distance_m=_round_float(sample.true_distance_m),
                weather=str(sample.weather),
                time_of_day=str(sample.time_of_day),
            )
        )
    return rows


def build_depth_representation_dataset(
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    depth_model: Any = None,
    strict: bool = True,
    sample_count_per_stratum: int = 2,
    sample_seed: int = 0,
    context_window_specs: Optional[Sequence[Dict[str, Any]]] = None,
    aggregation_methods: Optional[Sequence[str]] = None,
    inner_bbox_scale: float = 0.5,
    surrounding_bbox_scale: float = 1.8,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[DepthRepresentationRecord], List[SelectedSampleRecord]]:
    """Build the second-attempt depth-representation study dataset."""
    if depth_model is None:
        raise ValueError("depth_model must be provided to build the representation dataset.")

    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=strict)
    all_samples = list(loader.iter_samples())
    selected_samples = select_samples_per_stratum(
        all_samples,
        sample_count_per_stratum=sample_count_per_stratum,
        seed=sample_seed,
    )
    selected_sample_records = build_selected_sample_records(selected_samples)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "selection_complete",
                "num_available_samples": len(all_samples),
                "num_selected_samples": len(selected_samples),
                "num_strata": len({_sample_stratum_key(sample) for sample in all_samples}),
                "sample_count_per_stratum": int(sample_count_per_stratum),
                "sample_seed": int(sample_seed),
            }
        )

    resolved_context_specs = resolve_context_window_specs(context_window_specs)
    resolved_aggregation_methods = tuple(aggregation_methods or SUPPORTED_AGGREGATION_METHODS)
    for aggregation_method in resolved_aggregation_methods:
        if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                "Unsupported aggregation_method '{}'. Expected one of {}.".format(
                    aggregation_method,
                    SUPPORTED_AGGREGATION_METHODS,
                )
            )

    records: List[DepthRepresentationRecord] = []
    total_samples = len(selected_samples)
    total_contexts = len(resolved_context_specs)
    total_aggregations = len(resolved_aggregation_methods)
    for sample_index, (sample, selection_rank) in enumerate(selected_samples, start=1):
        image = loader.load_image(sample)
        stratum_key = "{}|{}|{}".format(
            _round_float(sample.true_distance_m),
            sample.weather,
            sample.time_of_day,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sample_start",
                    "sample_index": sample_index,
                    "total_samples": total_samples,
                    "stratum_key": stratum_key,
                    "image_path": str(sample.image_path),
                    "selection_rank_within_stratum": int(selection_rank),
                    "true_distance_m": _round_float(sample.true_distance_m),
                    "weather": str(sample.weather),
                    "time_of_day": str(sample.time_of_day),
                    "total_contexts": total_contexts,
                    "total_aggregations": total_aggregations,
                }
            )

        for context_index, context_spec in enumerate(resolved_context_specs, start=1):
            crop_data = extract_context_crop(
                image=image,
                bbox=sample.annotation.bbox,
                context_spec=context_spec,
            )
            depth_map = estimate_relative_depth(crop_data["image"], depth_model)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "context_complete",
                        "sample_index": sample_index,
                        "total_samples": total_samples,
                        "context_index": context_index,
                        "total_contexts": total_contexts,
                        "context_window": context_spec.name,
                        "image_path": str(sample.image_path),
                    }
                )

            for aggregation_method in resolved_aggregation_methods:
                representation = compute_depth_representation(
                    depth_map=depth_map,
                    bbox=crop_data["local_bbox"],
                    aggregation_method=aggregation_method,
                    inner_bbox_scale=inner_bbox_scale,
                    surrounding_bbox_scale=surrounding_bbox_scale,
                )
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_data["crop_bbox"]
                local_x1, local_y1, local_x2, local_y2 = crop_data["local_bbox"]
                focus_x1, focus_y1, focus_x2, focus_y2 = representation["focus_bbox"]
                ring_x1, ring_y1, ring_x2, ring_y2 = representation["surrounding_bbox"]

                records.append(
                    DepthRepresentationRecord(
                        image_path=str(sample.image_path),
                        label_path=str(sample.label_path),
                        stratum_key=stratum_key,
                        selection_rank_within_stratum=int(selection_rank),
                        true_distance_m=_round_float(sample.true_distance_m),
                        weather=str(sample.weather),
                        time_of_day=str(sample.time_of_day),
                        context_window=context_spec.name,
                        context_scale="full_image" if context_spec.scale is None else str(context_spec.scale),
                        aggregation_method=aggregation_method,
                        crop_x1=int(crop_x1),
                        crop_y1=int(crop_y1),
                        crop_x2=int(crop_x2),
                        crop_y2=int(crop_y2),
                        local_bbox_x1=int(local_x1),
                        local_bbox_y1=int(local_y1),
                        local_bbox_x2=int(local_x2),
                        local_bbox_y2=int(local_y2),
                        focus_bbox_x1=int(focus_x1),
                        focus_bbox_y1=int(focus_y1),
                        focus_bbox_x2=int(focus_x2),
                        focus_bbox_y2=int(focus_y2),
                        surrounding_bbox_x1=int(ring_x1),
                        surrounding_bbox_y1=int(ring_y1),
                        surrounding_bbox_x2=int(ring_x2),
                        surrounding_bbox_y2=int(ring_y2),
                        object_depth=_round_float(representation["object_depth"]),
                        object_depth_percentile_5_95=_round_float(
                            representation["object_depth_percentile_5_95"]
                        ),
                        surrounding_ring_depth=_round_float(representation["surrounding_ring_depth"]),
                        object_minus_ring=_round_float(representation["object_minus_ring"]),
                        object_minus_ring_normalized=_round_float(
                            representation["object_minus_ring_normalized"]
                        ),
                        robust_global_range=_round_float(representation["robust_global_range"]),
                        global_percentile_5=_round_float(representation["global_percentile_5"]),
                        global_percentile_95=_round_float(representation["global_percentile_95"]),
                    )
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sample_complete",
                    "sample_index": sample_index,
                    "total_samples": total_samples,
                    "records_written_so_far": len(records),
                    "image_path": str(sample.image_path),
                }
            )

    return records, selected_sample_records


def write_depth_representation_records_csv(
    records: Sequence[DepthRepresentationRecord],
    output_path: Union[str, Path],
) -> Path:
    """Write depth representation records to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEPTH_REPRESENTATION_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

    return path


def read_depth_representation_records_csv(
    input_path: Union[str, Path],
) -> List[DepthRepresentationRecord]:
    """Read depth representation records from CSV."""
    path = Path(input_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [DepthRepresentationRecord.from_dict(row) for row in reader]


def write_selected_sample_records_csv(
    records: Sequence[SelectedSampleRecord],
    output_path: Union[str, Path],
) -> Path:
    """Write the selected sample list for the study."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_SAMPLE_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

    return path


def summarize_depth_representation_records(
    records: Iterable[DepthRepresentationRecord],
    selected_samples: Iterable[SelectedSampleRecord],
) -> Dict[str, Any]:
    """Create a compact JSON summary for the second-attempt feature cache."""
    representation_rows = list(records)
    selected_rows = list(selected_samples)

    stratum_counts = Counter(record.stratum_key for record in selected_rows)
    weather_counts = Counter(record.weather for record in selected_rows)
    time_counts = Counter(record.time_of_day for record in selected_rows)
    distance_counts = Counter(record.true_distance_m for record in selected_rows)
    selection_rank_counts = Counter(record.selection_rank_within_stratum for record in selected_rows)
    context_counts = Counter(record.context_window for record in representation_rows)
    aggregation_counts = Counter(record.aggregation_method for record in representation_rows)

    return {
        "num_selected_samples": len(selected_rows),
        "num_representation_rows": len(representation_rows),
        "num_strata": len(stratum_counts),
        "mean_samples_per_stratum": (
            float(len(selected_rows)) / float(len(stratum_counts)) if stratum_counts else 0.0
        ),
        "samples_per_stratum": dict(stratum_counts),
        "selection_rank_counts": dict(sorted(selection_rank_counts.items())),
        "weather_counts": dict(weather_counts),
        "time_of_day_counts": dict(time_counts),
        "distance_counts": dict(sorted(distance_counts.items())),
        "context_window_counts": dict(context_counts),
        "aggregation_method_counts": dict(aggregation_counts),
        "score_fields": list(DEPTH_SCORE_FIELD_NAMES),
    }
