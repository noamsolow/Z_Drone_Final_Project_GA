"""
Study 03 for attempt3: all-images noisy-bbox aggregation study.

This study keeps the same two-stage random-forest idea as Study 02, but it
replaces the unrealistically exact bounding-box geometry with features built
from several small bbox perturbations. Depth features are also recomputed from
those perturbed boxes and then aggregated, so the pipeline is closer to a real
detector scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study01.run_study_01 import (
    BASE_CATEGORICAL_FEATURE_NAMES,
    _attach_prediction_feature,
    _compute_feature_correlations,
    _cross_fit_lower_model,
    _fit_random_forest_estimator,
    _load_or_create_splits,
    _prediction_rows,
    _select_rows,
    _write_csv_rows,
    _write_json,
)
from attempts.attempt3.studies.study02.run_study_02 import (
    _build_global_categorical_levels,
    _candidate_metric_row,
    _candidate_params_by_name,
    _evaluate_upper_rf_candidate,
    _fit_and_evaluate_selected_upper_rf,
    _select_best_candidate,
)
from pipeline.calibration.evaluation import build_grouped_error_summary
from pipeline.calibration.features import CONTINUOUS_FEATURE_NAMES
from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES
from pipeline.data.loader import DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import (
    SUPPORTED_AGGREGATION_METHODS,
    clamp_bbox,
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
META_COLUMNS = {
    "image_path",
    "label_path",
    "stratum_key",
    "true_distance_m",
    "weather",
    "time_of_day",
    "num_jitter_boxes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt3 Study 03 noisy-bbox aggregation study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 03 YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the planned workload without computing new noisy feature rows.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _normalize_path(path_value: str) -> str:
    return str(Path(path_value).resolve())


def _round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _sample_sort_key(sample: Any) -> Tuple[float, str, str, str]:
    return (
        float(sample.true_distance_m),
        str(sample.weather),
        str(sample.time_of_day),
        str(sample.image_path),
    )


def _resolve_context_and_aggregation_lists(study_config: Dict[str, Any]) -> Tuple[List[Any], List[str]]:
    context_specs = list(resolve_context_window_specs(study_config.get("context_windows")))
    aggregation_methods = list(study_config.get("aggregation_methods", list(SUPPORTED_AGGREGATION_METHODS)))
    for aggregation_method in aggregation_methods:
        if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                "Unsupported aggregation_method '{}'. Expected one of {}.".format(
                    aggregation_method,
                    SUPPORTED_AGGREGATION_METHODS,
                )
            )
    return context_specs, aggregation_methods


def _depth_aggregate_feature_names(
    context_specs: Sequence[Any],
    aggregation_methods: Sequence[str],
) -> Tuple[List[str], List[str]]:
    central: List[str] = []
    spread: List[str] = []
    for context_spec in context_specs:
        for aggregation_method in aggregation_methods:
            prefix = "{}__{}__object_depth".format(context_spec.name, aggregation_method)
            central.append(prefix + "__jitter_median")
            spread.append(prefix + "__jitter_std")
    return central, spread


def _geometry_aggregate_feature_names() -> Tuple[List[str], List[str]]:
    central = ["{}__jitter_median".format(name) for name in GEOMETRY_FEATURE_NAMES]
    spread = ["{}__jitter_std".format(name) for name in GEOMETRY_FEATURE_NAMES]
    return central, spread


def _build_expected_feature_layout(
    study_config: Dict[str, Any],
) -> Dict[str, Any]:
    context_specs, aggregation_methods = _resolve_context_and_aggregation_lists(study_config)
    depth_central_feature_names, depth_spread_feature_names = _depth_aggregate_feature_names(
        context_specs=context_specs,
        aggregation_methods=aggregation_methods,
    )
    geometry_central_feature_names, geometry_spread_feature_names = _geometry_aggregate_feature_names()
    raw_depth_feature_name = str(
        study_config.get(
            "raw_depth_feature_name",
            "bbox_only__inner50_median__object_depth__jitter_median",
        )
    )
    if raw_depth_feature_name not in depth_central_feature_names:
        raise ValueError(
            "raw_depth_feature_name '{}' is not one of the generated noisy depth features.".format(
                raw_depth_feature_name
            )
        )
    return {
        "context_specs": context_specs,
        "aggregation_methods": aggregation_methods,
        "depth_central_feature_names": depth_central_feature_names,
        "depth_spread_feature_names": depth_spread_feature_names,
        "geometry_central_feature_names": geometry_central_feature_names,
        "geometry_spread_feature_names": geometry_spread_feature_names,
        "raw_depth_feature_name": raw_depth_feature_name,
        "all_feature_names": depth_central_feature_names
        + depth_spread_feature_names
        + geometry_central_feature_names
        + geometry_spread_feature_names,
    }


def _resolve_jitter_specs(study_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_specs = list(study_config.get("jitter_boxes", []))
    if not raw_specs:
        raise ValueError("study.jitter_boxes must not be empty.")
    resolved: List[Dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs, start=1):
        resolved.append(
            {
                "name": str(raw_spec.get("name", "jitter_{}".format(index))),
                "dx_frac": float(raw_spec.get("dx_frac", 0.0)),
                "dy_frac": float(raw_spec.get("dy_frac", 0.0)),
                "scale": float(raw_spec.get("scale", 1.0)),
            }
        )
    return resolved


def _build_geometry_feature_dict(
    bbox: Sequence[int],
    image_width: int,
    image_height: int,
) -> Dict[str, float]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    bbox_width_px = int(x2 - x1)
    bbox_height_px = int(y2 - y1)
    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0
    return {
        "bbox_width_px": float(bbox_width_px),
        "bbox_height_px": float(bbox_height_px),
        "bbox_width_norm": float(bbox_width_px / float(image_width)),
        "bbox_height_norm": float(bbox_height_px / float(image_height)),
        "bbox_area_ratio": float((bbox_width_px * bbox_height_px) / float(image_width * image_height)),
        "bbox_aspect_ratio": float(bbox_width_px / float(bbox_height_px)),
        "bbox_center_x_norm": float(bbox_center_x / float(image_width)),
        "bbox_center_y_norm": float(bbox_center_y / float(image_height)),
    }


def _jitter_bbox(
    bbox: Sequence[int],
    image_width: int,
    image_height: int,
    jitter_spec: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    scale = float(jitter_spec.get("scale", 1.0))
    if scale <= 0.0:
        raise ValueError("Jitter scale must be positive.")

    shifted_center_x = center_x + float(jitter_spec.get("dx_frac", 0.0)) * width
    shifted_center_y = center_y + float(jitter_spec.get("dy_frac", 0.0)) * height
    jittered_width = max(1.0, width * scale)
    jittered_height = max(1.0, height * scale)

    return clamp_bbox(
        (
            shifted_center_x - jittered_width / 2.0,
            shifted_center_y - jittered_height / 2.0,
            shifted_center_x + jittered_width / 2.0,
            shifted_center_y + jittered_height / 2.0,
        ),
        image_width=image_width,
        image_height=image_height,
    )


def _translate_bbox_to_crop(
    global_bbox: Sequence[int],
    crop_bbox: Sequence[int],
    crop_width: int,
    crop_height: int,
) -> Tuple[int, int, int, int]:
    crop_x1, crop_y1, _, _ = [int(value) for value in crop_bbox]
    x1, y1, x2, y2 = [int(value) for value in global_bbox]
    return clamp_bbox(
        (
            x1 - crop_x1,
            y1 - crop_y1,
            x2 - crop_x1,
            y2 - crop_y1,
        ),
        image_width=crop_width,
        image_height=crop_height,
    )


def _aggregate_numeric_values(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot aggregate an empty value list.")
    return float(np.median(array)), float(np.std(array))


def _append_csv_row(row: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists or output_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _read_feature_table(
    input_path: Path,
    expected_feature_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not input_path.exists():
        return []
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return []
    missing_columns = [name for name in expected_feature_names if name not in reader.fieldnames]
    if missing_columns:
        raise ValueError(
            "Existing Study 03 feature table is missing expected columns: {}".format(missing_columns)
        )
    parsed_rows: List[Dict[str, Any]] = []
    numeric_columns = [name for name in reader.fieldnames if name not in META_COLUMNS]
    for row in rows:
        parsed = dict(row)
        parsed["true_distance_m"] = float(parsed["true_distance_m"])
        parsed["num_jitter_boxes"] = int(parsed["num_jitter_boxes"])
        for column_name in numeric_columns:
            parsed[column_name] = float(parsed[column_name])
        parsed_rows.append(parsed)
    return parsed_rows


def _write_feature_summary_checkpoint(
    output_path: Path,
    all_rows: Sequence[Dict[str, Any]],
    summary_payload: Dict[str, Any],
) -> None:
    weather_counts = Counter(str(row["weather"]) for row in all_rows)
    time_counts = Counter(str(row["time_of_day"]) for row in all_rows)
    summary = {
        "num_rows": len(all_rows),
        "weather_counts": dict(weather_counts),
        "time_of_day_counts": dict(time_counts),
        "unique_true_distances_m": sorted({float(row["true_distance_m"]) for row in all_rows}),
    }
    summary.update(summary_payload)
    _write_json(summary, output_path)


def _compute_noisy_feature_row(
    sample: Any,
    loader: DroneDatasetLoader,
    depth_model: Any,
    context_specs: Sequence[Any],
    aggregation_methods: Sequence[str],
    jitter_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    image = loader.load_image(sample)
    jittered_global_bboxes = [
        _jitter_bbox(
            bbox=sample.annotation.bbox,
            image_width=int(sample.image_width),
            image_height=int(sample.image_height),
            jitter_spec=jitter_spec,
        )
        for jitter_spec in jitter_specs
    ]

    row: Dict[str, Any] = {
        "image_path": str(sample.image_path),
        "label_path": str(sample.label_path),
        "stratum_key": "{}|{}|{}".format(
            _round_float(sample.true_distance_m),
            str(sample.weather),
            str(sample.time_of_day),
        ),
        "true_distance_m": _round_float(sample.true_distance_m),
        "weather": str(sample.weather),
        "time_of_day": str(sample.time_of_day),
        "num_jitter_boxes": int(len(jittered_global_bboxes)),
    }

    for feature_name in GEOMETRY_FEATURE_NAMES:
        jitter_values = [
            _build_geometry_feature_dict(
                bbox=bbox,
                image_width=int(sample.image_width),
                image_height=int(sample.image_height),
            )[feature_name]
            for bbox in jittered_global_bboxes
        ]
        feature_median, feature_std = _aggregate_numeric_values(jitter_values)
        row["{}__jitter_median".format(feature_name)] = _round_float(feature_median)
        row["{}__jitter_std".format(feature_name)] = _round_float(feature_std)

    for context_spec in context_specs:
        crop_data = extract_context_crop(
            image=image,
            bbox=sample.annotation.bbox,
            context_spec=context_spec,
        )
        depth_map = estimate_relative_depth(crop_data["image"], depth_model)
        crop_width = int(depth_map.shape[1])
        crop_height = int(depth_map.shape[0])
        local_jittered_bboxes = [
            _translate_bbox_to_crop(
                global_bbox=global_bbox,
                crop_bbox=crop_data["crop_bbox"],
                crop_width=crop_width,
                crop_height=crop_height,
            )
            for global_bbox in jittered_global_bboxes
        ]

        for aggregation_method in aggregation_methods:
            values: List[float] = []
            for jittered_local_bbox in local_jittered_bboxes:
                representation = compute_depth_representation(
                    depth_map=depth_map,
                    bbox=jittered_local_bbox,
                    aggregation_method=aggregation_method,
                )
                values.append(float(representation["object_depth"]))
            feature_median, feature_std = _aggregate_numeric_values(values)
            prefix = "{}__{}__object_depth".format(context_spec.name, aggregation_method)
            row[prefix + "__jitter_median"] = _round_float(feature_median)
            row[prefix + "__jitter_std"] = _round_float(feature_std)

    return row


def _build_or_resume_feature_table(
    resolved_config: Dict[str, Any],
    config_path: Path,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, Any]:
    output_root = _resolve_repo_path(str(resolved_config["output_root"]))
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_table_csv = features_dir / "noisy_aggregated_feature_table.csv"
    feature_summary_json = features_dir / "summary.json"
    selected_samples_csv = features_dir / "selected_samples.csv"

    dataset_root = _resolve_repo_path(str(resolved_config["dataset_root"]))
    strict = bool(resolved_config.get("strict", True))
    study_config = dict(resolved_config.get("study", {}))
    feature_layout = _build_expected_feature_layout(study_config)
    jitter_specs = _resolve_jitter_specs(study_config)

    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=strict)
    all_samples = sorted(list(loader.iter_samples()), key=_sample_sort_key)
    selected_rows = [
        {
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
            "true_distance_m": _round_float(sample.true_distance_m),
            "weather": str(sample.weather),
            "time_of_day": str(sample.time_of_day),
            "stratum_key": "{}|{}|{}".format(
                _round_float(sample.true_distance_m),
                str(sample.weather),
                str(sample.time_of_day),
            ),
        }
        for sample in all_samples
    ]
    _write_csv_rows(selected_rows, selected_samples_csv)

    existing_rows = _read_feature_table(
        input_path=feature_table_csv,
        expected_feature_names=feature_layout["all_feature_names"],
    )
    existing_by_path = {_normalize_path(row["image_path"]): row for row in existing_rows}
    missing_samples = [
        sample for sample in all_samples if _normalize_path(str(sample.image_path)) not in existing_by_path
    ]

    summary_payload = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "num_total_images": len(all_samples),
        "num_strata": len(
            {
                (
                    _round_float(sample.true_distance_m),
                    str(sample.weather),
                    str(sample.time_of_day),
                )
                for sample in all_samples
            }
        ),
        "selected_samples_csv": str(selected_samples_csv),
        "feature_table_csv": str(feature_table_csv),
        "depth_feature_count": len(feature_layout["depth_central_feature_names"]),
        "depth_spread_feature_count": len(feature_layout["depth_spread_feature_names"]),
        "geometry_feature_count": len(feature_layout["geometry_central_feature_names"]),
        "geometry_spread_feature_count": len(feature_layout["geometry_spread_feature_names"]),
        "contexts": [spec.name for spec in feature_layout["context_specs"]],
        "aggregation_methods": list(feature_layout["aggregation_methods"]),
        "jitter_box_names": [spec["name"] for spec in jitter_specs],
        "num_jitter_boxes": len(jitter_specs),
        "reused_rows": len(existing_rows),
        "missing_rows_to_compute": len(missing_samples),
        "total_expected_rows": len(all_samples),
    }

    if verbose:
        print(
            "[attempt3-study03] Noisy bbox feature setup\n"
            "  config: {}\n"
            "  dataset_root: {}\n"
            "  total_images: {}\n"
            "  total_contexts: {}\n"
            "  total_aggregations: {}\n"
            "  num_jitter_boxes: {}\n"
            "  reused_rows: {}\n"
            "  missing_rows_to_compute: {}".format(
                config_path,
                dataset_root,
                len(all_samples),
                len(feature_layout["context_specs"]),
                len(feature_layout["aggregation_methods"]),
                len(jitter_specs),
                len(existing_rows),
                len(missing_samples),
            )
        )

    if dry_run:
        all_rows = sorted(existing_rows, key=lambda row: _sample_sort_key(type("obj", (), row)))
        _write_feature_summary_checkpoint(feature_summary_json, all_rows, summary_payload)
        return {
            "feature_table_csv": str(feature_table_csv),
            "feature_summary_json": str(feature_summary_json),
            "all_rows": all_rows,
            "feature_layout": feature_layout,
        }

    depth_model = None
    if missing_samples:
        depth_config = dict(resolved_config.get("depth_model", {}))
        if verbose:
            print(
                "[attempt3-study03] Loading depth model\n"
                "  model_name: {}\n"
                "  device: {}\n"
                "  local_files_only: {}".format(
                    depth_config.get("model_name"),
                    depth_config.get("device"),
                    bool(depth_config.get("local_files_only", False)),
                )
            )
        depth_model = load_depth_anything_v2(
            model_name=str(depth_config.get("model_name")) if depth_config.get("model_name") else None,
            device=depth_config.get("device"),
            local_files_only=bool(depth_config.get("local_files_only", False)),
        )

    progress_log_every_samples = max(1, int(study_config.get("progress_log_every_samples", 25)))
    all_rows = list(existing_rows)
    for sample_index, sample in enumerate(missing_samples, start=1):
        if verbose and (
            sample_index == 1
            or sample_index == len(missing_samples)
            or sample_index % progress_log_every_samples == 0
        ):
            print(
                "[attempt3-study03] Sample {}/{}\n"
                "  distance: {} m\n"
                "  weather: {}\n"
                "  time_of_day: {}\n"
                "  image: {}".format(
                    sample_index,
                    len(missing_samples),
                    _round_float(sample.true_distance_m),
                    str(sample.weather),
                    str(sample.time_of_day),
                    sample.image_path,
                )
            )

        row = _compute_noisy_feature_row(
            sample=sample,
            loader=loader,
            depth_model=depth_model,
            context_specs=feature_layout["context_specs"],
            aggregation_methods=feature_layout["aggregation_methods"],
            jitter_specs=jitter_specs,
        )
        _append_csv_row(row, feature_table_csv)
        all_rows.append(row)
        if verbose and (
            sample_index == 1
            or sample_index == len(missing_samples)
            or sample_index % progress_log_every_samples == 0
        ):
            print(
                "[attempt3-study03]   saved noisy feature row | rows now {}".format(len(all_rows))
            )
        _write_feature_summary_checkpoint(feature_summary_json, all_rows, summary_payload)

    all_rows = sorted(
        all_rows,
        key=lambda row: (
            float(row["true_distance_m"]),
            str(row["weather"]),
            str(row["time_of_day"]),
            str(row["image_path"]),
        ),
    )
    _write_csv_rows(all_rows, feature_table_csv)
    _write_feature_summary_checkpoint(feature_summary_json, all_rows, summary_payload)
    return {
        "feature_table_csv": str(feature_table_csv),
        "feature_summary_json": str(feature_summary_json),
        "all_rows": all_rows,
        "feature_layout": feature_layout,
    }


def _build_model_specs(
    config: Dict[str, Any],
    feature_layout: Dict[str, Any],
    lower_prediction_feature_name: str,
) -> List[Dict[str, Any]]:
    geometry_feature_names = list(feature_layout["geometry_central_feature_names"]) + list(
        feature_layout["geometry_spread_feature_names"]
    )
    raw_depth_feature_name = str(feature_layout["raw_depth_feature_name"])
    candidate_name_map = {
        str(key): [str(value) for value in values]
        for key, values in dict(config.get("upper_random_forest", {}).get("model_candidate_names", {})).items()
    }
    all_specs: Dict[str, Dict[str, Any]] = {
        "raw_depth_only_rf": {
            "model_name": "raw_depth_only_rf",
            "continuous_feature_names": [raw_depth_feature_name],
            "categorical_feature_names": [],
            "uses_lower_model": False,
        },
        "geometry_metadata_rf": {
            "model_name": "geometry_metadata_rf",
            "continuous_feature_names": geometry_feature_names,
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "raw_depth_geometry_metadata_rf": {
            "model_name": "raw_depth_geometry_metadata_rf",
            "continuous_feature_names": [raw_depth_feature_name] + geometry_feature_names,
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "small_rf_depth_only_rf": {
            "model_name": "small_rf_depth_only_rf",
            "continuous_feature_names": [lower_prediction_feature_name],
            "categorical_feature_names": [],
            "uses_lower_model": True,
        },
        "stacked_rf_depth_geometry_metadata_rf": {
            "model_name": "stacked_rf_depth_geometry_metadata_rf",
            "continuous_feature_names": [lower_prediction_feature_name] + geometry_feature_names,
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": True,
        },
    }
    requested_names = list(config.get("modeling", {}).get("model_names", all_specs.keys()))
    missing = [name for name in requested_names if name not in all_specs]
    if missing:
        raise ValueError("Unsupported model names in config: {}".format(missing))
    model_specs: List[Dict[str, Any]] = []
    for name in requested_names:
        spec = dict(all_specs[name])
        spec["candidate_names"] = list(candidate_name_map.get(name, []))
        model_specs.append(spec)
    return model_specs


def _prepare_outer_fold_caches(
    merged_rows: Sequence[Dict[str, Any]],
    cv_folds: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    lower_prediction_feature_name: str,
    lower_rf_params: Dict[str, Any],
    lower_inner_num_folds: int,
    lower_inner_seed: int,
    needs_lower_model: bool,
    verbose: bool,
) -> Dict[int, Dict[str, Any]]:
    caches: Dict[int, Dict[str, Any]] = {}
    for outer_fold_index, fold in enumerate(cv_folds, start=1):
        train_rows = _select_rows(merged_rows, fold["train_indices"])
        val_rows = _select_rows(merged_rows, fold["val_indices"])
        cache: Dict[str, Any] = {
            "fold_index": int(fold["fold_index"]),
            "train_base_rows": train_rows,
            "val_base_rows": val_rows,
            "global_val_indices": [int(index) for index in fold["val_indices"]],
        }
        if needs_lower_model:
            if verbose:
                print(
                    "[attempt3-study03] Preparing lower-model cache for outer fold {}/{} | train_rows={} | val_rows={}".format(
                        outer_fold_index,
                        len(cv_folds),
                        len(train_rows),
                        len(val_rows),
                    )
                )
            lower_cache = _cross_fit_lower_model(
                train_rows=train_rows,
                lower_feature_names=lower_feature_names,
                rf_params=lower_rf_params,
                num_folds=lower_inner_num_folds,
                seed=lower_inner_seed + int(fold["fold_index"]),
                verbose=verbose,
                progress_label="study03 outer_fold={}".format(outer_fold_index),
            )
            train_augmented = _attach_prediction_feature(
                train_rows,
                lower_prediction_feature_name,
                lower_cache["oof_predictions"],
            )
            estimator = lower_cache["full_estimator"]
            val_lower_predictions = estimator.predict(
                np.asarray(
                    [[float(row[name]) for name in lower_feature_names] for row in val_rows],
                    dtype=np.float64,
                )
            )
            val_augmented = _attach_prediction_feature(
                val_rows,
                lower_prediction_feature_name,
                val_lower_predictions,
            )
            cache["train_lower_rows"] = train_augmented
            cache["val_lower_rows"] = val_augmented
        caches[int(fold["fold_index"])] = cache
    return caches


def _prepare_heldout_lower_cache(
    dev_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    lower_prediction_feature_name: str,
    lower_rf_params: Dict[str, Any],
    lower_inner_num_folds: int,
    lower_inner_seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    if verbose:
        print(
            "[attempt3-study03] Building shared lower-model feature for held-out test | dev_rows={} | test_rows={}".format(
                len(dev_rows),
                len(test_rows),
            )
        )
    lower_cache = _cross_fit_lower_model(
        train_rows=dev_rows,
        lower_feature_names=lower_feature_names,
        rf_params=lower_rf_params,
        num_folds=lower_inner_num_folds,
        seed=lower_inner_seed + 1000,
        verbose=verbose,
        progress_label="study03 heldout-test-prep",
    )
    dev_augmented_rows = _attach_prediction_feature(
        dev_rows,
        lower_prediction_feature_name,
        lower_cache["oof_predictions"],
    )
    estimator = lower_cache["full_estimator"]
    test_lower_predictions = estimator.predict(
        np.asarray(
            [[float(row[name]) for name in lower_feature_names] for row in test_rows],
            dtype=np.float64,
        )
    )
    test_augmented_rows = _attach_prediction_feature(
        test_rows,
        lower_prediction_feature_name,
        test_lower_predictions,
    )
    return {
        "dev_lower_rows": dev_augmented_rows,
        "test_lower_rows": test_augmented_rows,
        "test_lower_predictions": test_lower_predictions,
        "feature_importance_rows": list(lower_cache["feature_importance_rows"]),
    }


def run_study(
    config_path: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    resolved_config = load_config(config_path)
    feature_build = _build_or_resume_feature_table(
        resolved_config=resolved_config,
        config_path=config_path,
        dry_run=dry_run,
        verbose=verbose,
    )
    merged_rows = list(feature_build["all_rows"])
    feature_layout = dict(feature_build["feature_layout"])
    output_root = _resolve_repo_path(str(resolved_config["output_root"]))
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    feature_correlations = _compute_feature_correlations(
        merged_rows,
        continuous_feature_names=list(feature_layout["all_feature_names"]),
    )
    _write_csv_rows(feature_correlations, reports_dir / "feature_correlations.csv")

    lower_prediction_feature_name = str(
        resolved_config.get("lower_model", {}).get("score_feature_name", "depth_lower_model_prediction")
    )
    selected_lower_model_name = "study03_noisy_small_rf"
    selected_lower_feature_names = list(feature_layout["depth_central_feature_names"])
    model_specs = _build_model_specs(
        config=resolved_config,
        feature_layout=feature_layout,
        lower_prediction_feature_name=lower_prediction_feature_name,
    )
    needs_lower_model = any(bool(spec["uses_lower_model"]) for spec in model_specs)

    if dry_run:
        summary = {
            "study_name": str(resolved_config.get("study_name", "study03")),
            "num_rows": len(merged_rows),
            "num_lower_features": len(selected_lower_feature_names),
            "raw_depth_feature_name": str(feature_layout["raw_depth_feature_name"]),
            "model_names": [str(spec["model_name"]) for spec in model_specs],
            "feature_table_csv": feature_build["feature_table_csv"],
        }
        _write_json(summary, reports_dir / "summary.json")
        return summary

    split_config = dict(resolved_config.get("split", {}))
    split_data = _load_or_create_splits(
        rows=merged_rows,
        split_dir=split_dir,
        holdout_fraction=float(split_config.get("holdout_fraction", 0.15)),
        num_folds=int(split_config.get("num_folds", 5)),
        seed=int(split_config.get("seed", 0)),
    )
    dev_rows = _select_rows(merged_rows, split_data["dev_indices"])
    test_rows = _select_rows(merged_rows, split_data["test_indices"])
    global_categorical_levels = _build_global_categorical_levels(
        rows=merged_rows,
        categorical_feature_names=BASE_CATEGORICAL_FEATURE_NAMES,
    )

    lower_model_config = dict(resolved_config.get("lower_model", {}))
    lower_inner_num_folds = int(lower_model_config.get("inner_num_folds", 5))
    lower_inner_seed = int(lower_model_config.get("inner_seed", 0))
    lower_rf_params = dict(lower_model_config.get("random_forest", {}))

    outer_fold_caches = _prepare_outer_fold_caches(
        merged_rows=merged_rows,
        cv_folds=split_data["cv_folds"],
        lower_feature_names=selected_lower_feature_names,
        lower_prediction_feature_name=lower_prediction_feature_name,
        lower_rf_params=lower_rf_params,
        lower_inner_num_folds=lower_inner_num_folds,
        lower_inner_seed=lower_inner_seed,
        needs_lower_model=needs_lower_model,
        verbose=verbose,
    )

    upper_candidate_params = _candidate_params_by_name(resolved_config)
    cv_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "models": {},
    }
    upper_candidate_rows: List[Dict[str, Any]] = []
    selected_model_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    cv_prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []
    selected_candidates: Dict[str, Dict[str, Any]] = {}

    for model_index, model_spec in enumerate(model_specs, start=1):
        if verbose:
            print(
                "[attempt3-study03] CV evaluation {}/{} | {} | uses_lower_model={}".format(
                    model_index,
                    len(model_specs),
                    model_spec["model_name"],
                    bool(model_spec["uses_lower_model"]),
                )
            )
        candidate_results: List[Dict[str, Any]] = []
        for candidate_index, candidate_name in enumerate(model_spec["candidate_names"], start=1):
            candidate_params = upper_candidate_params[str(candidate_name)]
            if verbose:
                print(
                    "[attempt3-study03] Evaluating candidate {}/{} | {} | {}".format(
                        candidate_index,
                        len(model_spec["candidate_names"]),
                        model_spec["model_name"],
                        candidate_name,
                    )
                )
            result = _evaluate_upper_rf_candidate(
                model_spec=model_spec,
                candidate_name=str(candidate_name),
                candidate_params=candidate_params,
                outer_fold_caches=outer_fold_caches,
                cv_folds=split_data["cv_folds"],
                dev_indices=split_data["dev_indices"],
                global_categorical_levels=global_categorical_levels,
                verbose=verbose,
            )
            candidate_results.append(result)
            upper_candidate_rows.append(_candidate_metric_row(result))

        best_candidate = _select_best_candidate(candidate_results)
        selected_candidates[str(model_spec["model_name"])] = best_candidate
        selected_summary = dict(best_candidate["summary"])
        selected_model_rows.append(
            {
                "model_name": str(model_spec["model_name"]),
                "selected_candidate_name": str(best_candidate["candidate_name"]),
                "num_features": len(best_candidate["rf_feature_names"]),
                "feature_names": " | ".join(best_candidate["rf_feature_names"]),
                "oof_mae": float(best_candidate["oof_overall"]["mae"]),
                "oof_mean_relative_error": float(best_candidate["oof_overall"]["mean_relative_error"]),
                "oof_rmse": float(best_candidate["oof_overall"]["rmse"]),
                "oof_r2": float(best_candidate["oof_overall"]["r2"]),
                **selected_summary,
                **{"param_{}".format(key): value for key, value in best_candidate["candidate_params"].items()},
            }
        )
        cv_metrics["models"][str(model_spec["model_name"])] = {
            "selected_candidate_name": str(best_candidate["candidate_name"]),
            "selected_candidate_params": dict(best_candidate["candidate_params"]),
            "candidate_metric_rows": [_candidate_metric_row(result) for result in candidate_results],
            "fold_metrics": best_candidate["fold_metrics"],
            "summary": dict(best_candidate["summary"]),
            "oof_overall": dict(best_candidate["oof_overall"]),
            "oof_per_distance_range": dict(best_candidate["oof_per_distance_range"]),
        }
        ranking_rows.append(
            {
                "model_name": str(model_spec["model_name"]),
                "selected_candidate_name": str(best_candidate["candidate_name"]),
                "oof_mae": float(best_candidate["oof_overall"]["mae"]),
                "oof_mean_relative_error": float(best_candidate["oof_overall"]["mean_relative_error"]),
            }
        )
        cv_prediction_rows.extend(best_candidate["oof_prediction_rows"])
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=best_candidate["oof_rows"],
                predictions=best_candidate["oof_predictions"],
                model_name="{}__{}".format(model_spec["model_name"], best_candidate["candidate_name"]),
                split_name="cv_oof",
            )
        )
        if verbose:
            print(
                "[attempt3-study03] Selected best candidate | {} | {} | oof_mae={:.4f}".format(
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                    float(best_candidate["oof_overall"]["mae"]),
                )
            )

    ranked_models = sorted(
        ranking_rows,
        key=lambda row: (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for rank_index, row in enumerate(ranked_models, start=1):
        row["rank"] = rank_index
    winner_model_name = str(ranked_models[0]["model_name"])
    winner_candidate_name = str(ranked_models[0]["selected_candidate_name"])
    if verbose:
        print(
            "[attempt3-study03] CV winner selected | {} | {} | oof_mae={:.4f}".format(
                winner_model_name,
                winner_candidate_name,
                float(ranked_models[0]["oof_mae"]),
            )
        )
    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["winner_candidate_name"] = winner_candidate_name
    cv_metrics["ranking"] = ranked_models
    _write_json(cv_metrics, reports_dir / "cv_metrics.json")
    _write_csv_rows(upper_candidate_rows, reports_dir / "upper_rf_candidate_metrics.csv")
    _write_csv_rows(selected_model_rows, reports_dir / "upper_rf_selected_models.csv")
    _write_csv_rows(cv_prediction_rows, reports_dir / "cv_predictions.csv")

    heldout_lower_cache: Dict[str, Any]
    if needs_lower_model:
        heldout_lower_cache = _prepare_heldout_lower_cache(
            dev_rows=dev_rows,
            test_rows=test_rows,
            lower_feature_names=selected_lower_feature_names,
            lower_prediction_feature_name=lower_prediction_feature_name,
            lower_rf_params=lower_rf_params,
            lower_inner_num_folds=lower_inner_num_folds,
            lower_inner_seed=lower_inner_seed,
            verbose=verbose,
        )
    else:
        heldout_lower_cache = {
            "dev_lower_rows": dev_rows,
            "test_lower_rows": test_rows,
            "test_lower_predictions": None,
            "feature_importance_rows": [],
        }

    test_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "winner_model_name": winner_model_name,
        "winner_candidate_name": winner_candidate_name,
        "selected_lower_model_name": selected_lower_model_name,
        "models": {},
    }
    test_prediction_rows: List[Dict[str, Any]] = []
    upper_feature_importance_rows: List[Dict[str, Any]] = []

    for model_index, model_spec in enumerate(model_specs, start=1):
        best_candidate = selected_candidates[str(model_spec["model_name"])]
        if verbose:
            print(
                "[attempt3-study03] Held-out test fit {}/{} | {} | candidate={}".format(
                    model_index,
                    len(model_specs),
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                )
            )
        train_rows = heldout_lower_cache["dev_lower_rows"] if model_spec["uses_lower_model"] else dev_rows
        eval_rows = heldout_lower_cache["test_lower_rows"] if model_spec["uses_lower_model"] else test_rows
        test_result = _fit_and_evaluate_selected_upper_rf(
            model_spec=model_spec,
            selected_candidate=best_candidate,
            train_rows=train_rows,
            test_rows=eval_rows,
            global_categorical_levels=global_categorical_levels,
        )
        test_metrics["models"][str(model_spec["model_name"])] = {
            "selected_candidate_name": str(best_candidate["candidate_name"]),
            "selected_candidate_params": dict(best_candidate["candidate_params"]),
            "is_cv_winner": str(model_spec["model_name"]) == winner_model_name,
            "overall": dict(test_result["metrics"]["overall"]),
            "per_distance_range": dict(test_result["metrics"]["per_distance_range"]),
        }
        test_prediction_rows.extend(test_result["prediction_rows"])
        upper_feature_importance_rows.extend(test_result["importance_rows"])
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=eval_rows,
                predictions=test_result["predictions"],
                model_name="{}__{}".format(model_spec["model_name"], best_candidate["candidate_name"]),
                split_name="test",
            )
        )
        if verbose:
            print(
                "[attempt3-study03] Completed held-out test fit | {} | candidate={} | test_mae={:.4f}".format(
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                    float(test_result["metrics"]["overall"]["mae"]),
                )
            )

    _write_json(test_metrics, reports_dir / "test_metrics.json")
    _write_csv_rows(test_prediction_rows, reports_dir / "test_predictions.csv")
    _write_csv_rows(grouped_summary_rows, reports_dir / "grouped_error_summary.csv")
    _write_csv_rows(upper_feature_importance_rows, reports_dir / "upper_rf_feature_importances.csv")
    _write_csv_rows(
        list(heldout_lower_cache.get("feature_importance_rows", [])),
        reports_dir / "lower_model_feature_importances.csv",
    )
    _write_json(
        {
            "selected_lower_model_name": selected_lower_model_name,
            "selected_lower_feature_names": list(selected_lower_feature_names),
            "num_selected_lower_features": len(selected_lower_feature_names),
            "random_forest_params": lower_rf_params,
            "raw_depth_feature_name": str(feature_layout["raw_depth_feature_name"]),
            "depth_spread_feature_names": list(feature_layout["depth_spread_feature_names"]),
        },
        reports_dir / "lower_model_summary.json",
    )

    winner_test_metrics = test_metrics["models"][winner_model_name]["overall"]
    selected_candidate_map = {
        model_name: selected_candidates[model_name]["candidate_name"] for model_name in selected_candidates
    }
    did_unbounded_win = any(str(candidate_name) == "rf_unbounded" for candidate_name in selected_candidate_map.values())
    report_summary = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "num_rows": len(merged_rows),
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "num_selected_lower_features": len(selected_lower_feature_names),
        "raw_depth_feature_name": str(feature_layout["raw_depth_feature_name"]),
        "winner_model_name": winner_model_name,
        "winner_candidate_name": winner_candidate_name,
        "winner_cv_oof_mae": float(ranked_models[0]["oof_mae"]),
        "winner_cv_oof_mean_relative_error": float(ranked_models[0]["oof_mean_relative_error"]),
        "winner_test_mae": float(winner_test_metrics["mae"]),
        "winner_test_mean_relative_error": float(winner_test_metrics["mean_relative_error"]),
        "selected_upper_candidates_by_model": selected_candidate_map,
        "did_unbounded_depth_win_any_model": bool(did_unbounded_win),
        "feature_table_csv": feature_build["feature_table_csv"],
        "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
        "test_metrics_json": str(reports_dir / "test_metrics.json"),
        "upper_rf_candidate_metrics_csv": str(reports_dir / "upper_rf_candidate_metrics.csv"),
    }
    _write_json(report_summary, reports_dir / "summary.json")

    if verbose:
        print(
            "[attempt3-study03] Study complete\n"
            "  winner_model: {}\n"
            "  winner_candidate: {}\n"
            "  winner_test_mae: {:.4f}\n"
            "  summary_json: {}".format(
                winner_model_name,
                winner_candidate_name,
                float(report_summary["winner_test_mae"]),
                reports_dir / "summary.json",
            )
        )

    return report_summary


def main() -> None:
    args = parse_args()
    run_study(config_path=args.config, dry_run=bool(args.dry_run), verbose=True)


if __name__ == "__main__":
    main()
