"""
Study 02 for attempt4: realistic noisy-bbox rows with XGBoost models.

This study keeps the Study 03 detector-like bbox perturbations, but it does
not aggregate them back into one robust row. Instead, each jittered bbox is
treated as a separate training/evaluation row. To prevent leakage, all jitter
rows derived from the same source image are split together as one group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
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
    _build_feature_matrix,
    _compute_feature_correlations,
    _cv_summary_from_fold_rows,
    _prediction_rows,
    _select_rows,
    _write_csv_rows,
    _write_json,
)
from attempts.attempt3.studies.study02.run_study_02 import (
    _build_global_categorical_levels,
    _build_rf_feature_names as _build_tabular_feature_names,
    _build_rf_matrix as _build_tabular_matrix,
)
from attempts.attempt3.studies.study03.run_study_03 import (
    _append_csv_row,
    _build_geometry_feature_dict,
    _jitter_bbox,
    _resolve_jitter_specs,
    _resolve_repo_path,
    _round_float,
    _sample_sort_key,
    _write_feature_summary_checkpoint,
)
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions
from pipeline.calibration.splits import make_distance_range_label, make_stratified_cv_folds
from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES
from pipeline.data.loader import DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import (
    SUPPORTED_AGGREGATION_METHODS,
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
META_COLUMNS = {
    "image_path",
    "label_path",
    "source_image_path",
    "source_label_path",
    "source_group_id",
    "jitter_name",
    "augmented_row_id",
    "true_distance_m",
    "weather",
    "time_of_day",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attempt4 Study 02 noisy-bbox expanded-row XGBoost study."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 02 YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the planned workload without computing new jitter rows.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_path(path_value: str) -> str:
    return str(Path(path_value).resolve())


def _resolve_context_and_aggregation_lists(
    study_config: Dict[str, Any],
) -> Tuple[List[Any], List[str]]:
    context_specs = list(resolve_context_window_specs(study_config.get("context_windows")))
    aggregation_methods = list(
        study_config.get("aggregation_methods", list(SUPPORTED_AGGREGATION_METHODS))
    )
    for aggregation_method in aggregation_methods:
        if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                "Unsupported aggregation_method '{}'. Expected one of {}.".format(
                    aggregation_method,
                    SUPPORTED_AGGREGATION_METHODS,
                )
            )
    return context_specs, aggregation_methods


def _depth_feature_names(
    context_specs: Sequence[Any],
    aggregation_methods: Sequence[str],
) -> List[str]:
    output: List[str] = []
    for context_spec in context_specs:
        for aggregation_method in aggregation_methods:
            output.append("{}__{}__object_depth".format(context_spec.name, aggregation_method))
    return output


def _build_expected_feature_layout(study_config: Dict[str, Any]) -> Dict[str, Any]:
    context_specs, aggregation_methods = _resolve_context_and_aggregation_lists(study_config)
    depth_feature_names = _depth_feature_names(context_specs, aggregation_methods)
    geometry_feature_names = list(GEOMETRY_FEATURE_NAMES)
    raw_depth_feature_name = str(
        study_config.get("raw_depth_feature_name", "bbox_only__inner50_median__object_depth")
    )
    if raw_depth_feature_name not in depth_feature_names:
        raise ValueError(
            "raw_depth_feature_name '{}' is not one of the generated depth features.".format(
                raw_depth_feature_name
            )
        )
    return {
        "context_specs": context_specs,
        "aggregation_methods": aggregation_methods,
        "depth_feature_names": depth_feature_names,
        "geometry_feature_names": geometry_feature_names,
        "raw_depth_feature_name": raw_depth_feature_name,
        "all_feature_names": depth_feature_names + geometry_feature_names,
    }


def _make_augmented_row_id(source_image_path: str, jitter_name: str) -> str:
    return "{}::{}".format(_normalize_path(source_image_path), str(jitter_name))


def _compute_jittered_feature_rows(
    sample: Any,
    loader: DroneDatasetLoader,
    depth_model: Any,
    context_specs: Sequence[Any],
    aggregation_methods: Sequence[str],
    jitter_specs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    image = loader.load_image(sample)
    output_rows: List[Dict[str, Any]] = []

    for jitter_spec in jitter_specs:
        jitter_name = str(jitter_spec["name"])
        jittered_global_bbox = _jitter_bbox(
            bbox=sample.annotation.bbox,
            image_width=int(sample.image_width),
            image_height=int(sample.image_height),
            jitter_spec=jitter_spec,
        )
        synthetic_image_path = "{}::jitter={}".format(str(sample.image_path), jitter_name)
        row: Dict[str, Any] = {
            "image_path": synthetic_image_path,
            "label_path": str(sample.label_path),
            "source_image_path": str(sample.image_path),
            "source_label_path": str(sample.label_path),
            "source_group_id": _normalize_path(str(sample.image_path)),
            "jitter_name": jitter_name,
            "augmented_row_id": _make_augmented_row_id(str(sample.image_path), jitter_name),
            "true_distance_m": _round_float(sample.true_distance_m),
            "weather": str(sample.weather),
            "time_of_day": str(sample.time_of_day),
        }

        geometry_features = _build_geometry_feature_dict(
            bbox=jittered_global_bbox,
            image_width=int(sample.image_width),
            image_height=int(sample.image_height),
        )
        for feature_name in GEOMETRY_FEATURE_NAMES:
            row[feature_name] = _round_float(geometry_features[feature_name])

        for context_spec in context_specs:
            crop_data = extract_context_crop(
                image=image,
                bbox=jittered_global_bbox,
                context_spec=context_spec,
            )
            depth_map = estimate_relative_depth(crop_data["image"], depth_model)
            local_bbox = crop_data["local_bbox"]
            for aggregation_method in aggregation_methods:
                representation = compute_depth_representation(
                    depth_map=depth_map,
                    bbox=local_bbox,
                    aggregation_method=aggregation_method,
                )
                feature_name = "{}__{}__object_depth".format(
                    context_spec.name,
                    aggregation_method,
                )
                row[feature_name] = _round_float(representation["object_depth"])

        output_rows.append(row)

    return output_rows


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
            "Existing attempt4 Study 02 feature table is missing expected columns: {}".format(missing_columns)
        )
    parsed_rows: List[Dict[str, Any]] = []
    numeric_columns = [name for name in reader.fieldnames if name not in META_COLUMNS]
    for row in rows:
        parsed = dict(row)
        parsed["true_distance_m"] = float(parsed["true_distance_m"])
        for column_name in numeric_columns:
            parsed[column_name] = float(parsed[column_name])
        parsed_rows.append(parsed)
    return parsed_rows


def _validate_existing_feature_rows(
    existing_rows: Sequence[Dict[str, Any]],
    expected_source_ids: Sequence[str],
    expected_jitter_names: Sequence[str],
    total_expected_rows: int,
    feature_table_csv: Path,
) -> None:
    if not existing_rows:
        return

    expected_source_id_set = {str(source_id) for source_id in expected_source_ids}
    expected_jitter_name_set = {str(jitter_name) for jitter_name in expected_jitter_names}
    seen_row_ids: set[str] = set()
    unexpected_jitter_names: set[str] = set()
    unexpected_source_ids: set[str] = set()
    jitter_names_by_source: Dict[str, set[str]] = defaultdict(set)

    for row in existing_rows:
        row_id = str(row["augmented_row_id"])
        if row_id in seen_row_ids:
            raise ValueError(
                "Existing attempt4 Study 02 feature table contains duplicate augmented_row_id values. "
                "Delete '{}' and rerun the study.".format(feature_table_csv)
            )
        seen_row_ids.add(row_id)

        source_id = str(row["source_group_id"])
        jitter_name = str(row["jitter_name"])
        if source_id not in expected_source_id_set:
            unexpected_source_ids.add(source_id)
        if jitter_name not in expected_jitter_name_set:
            unexpected_jitter_names.add(jitter_name)
        jitter_names_by_source[source_id].add(jitter_name)

    overfilled_sources = [
        source_id
        for source_id, source_jitter_names in jitter_names_by_source.items()
        if len(source_jitter_names) > len(expected_jitter_name_set)
    ]

    incompatibility_messages: List[str] = []
    if len(existing_rows) > total_expected_rows:
        incompatibility_messages.append(
            "row count {} exceeds expected {}".format(len(existing_rows), total_expected_rows)
        )
    if unexpected_jitter_names:
        incompatibility_messages.append(
            "unexpected jitter names: {}".format(", ".join(sorted(unexpected_jitter_names)))
        )
    if unexpected_source_ids:
        incompatibility_messages.append(
            "unexpected source ids: {}".format(len(unexpected_source_ids))
        )
    if overfilled_sources:
        incompatibility_messages.append(
            "some source images already have more jitter rows than the current config allows"
        )

    if incompatibility_messages:
        raise ValueError(
            "Existing attempt4 Study 02 feature table is incompatible with the current config ({}) . "
            "Delete '{}' and rerun the study.".format(
                "; ".join(incompatibility_messages),
                feature_table_csv,
            )
        )


def _build_or_resume_feature_table(
    resolved_config: Dict[str, Any],
    config_path: Path,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, Any]:
    output_root = _resolve_repo_path(str(resolved_config["output_root"]))
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_table_csv = features_dir / "noisy_expanded_feature_table.csv"
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
            "source_group_id": _normalize_path(str(sample.image_path)),
        }
        for sample in all_samples
    ]
    _write_csv_rows(selected_rows, selected_samples_csv)

    feature_table_seed_csv = resolved_config.get("feature_table_seed_csv")
    if feature_table_seed_csv and not feature_table_csv.exists():
        seed_feature_table_csv = _resolve_repo_path(str(feature_table_seed_csv))
        seeded_rows = _read_feature_table(
            input_path=seed_feature_table_csv,
            expected_feature_names=feature_layout["all_feature_names"],
        )
        if seeded_rows:
            if verbose:
                print(
                    "[attempt4-study02] Seeding feature table from {}\n"
                    "  target: {}\n"
                    "  seeded_rows: {}".format(
                        seed_feature_table_csv,
                        feature_table_csv,
                        len(seeded_rows),
                    )
                )
            _write_csv_rows(seeded_rows, feature_table_csv)

    existing_rows = _read_feature_table(
        input_path=feature_table_csv,
        expected_feature_names=feature_layout["all_feature_names"],
    )
    expected_source_ids = [_normalize_path(str(sample.image_path)) for sample in all_samples]
    total_expected_rows = len(all_samples) * len(jitter_specs)
    _validate_existing_feature_rows(
        existing_rows=existing_rows,
        expected_source_ids=expected_source_ids,
        expected_jitter_names=[spec["name"] for spec in jitter_specs],
        total_expected_rows=total_expected_rows,
        feature_table_csv=feature_table_csv,
    )
    existing_ids = {str(row["augmented_row_id"]) for row in existing_rows}

    missing_row_ids = []
    for sample in all_samples:
        for jitter_spec in jitter_specs:
            row_id = _make_augmented_row_id(str(sample.image_path), str(jitter_spec["name"]))
            if row_id not in existing_ids:
                missing_row_ids.append(row_id)

    sample_ids_needing_compute = {
        row_id.split("::", 1)[0] for row_id in missing_row_ids
    }
    missing_samples = [
        sample
        for sample in all_samples
        if _normalize_path(str(sample.image_path)) in sample_ids_needing_compute
    ]

    summary_payload = {
        "study_name": str(resolved_config.get("study_name", "study02")),
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "num_total_images": len(all_samples),
        "num_total_augmented_rows": total_expected_rows,
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
        "depth_feature_count": len(feature_layout["depth_feature_names"]),
        "geometry_feature_count": len(feature_layout["geometry_feature_names"]),
        "contexts": [spec.name for spec in feature_layout["context_specs"]],
        "aggregation_methods": list(feature_layout["aggregation_methods"]),
        "jitter_box_names": [spec["name"] for spec in jitter_specs],
        "num_jitter_boxes": len(jitter_specs),
        "reused_rows": len(existing_rows),
        "missing_rows_to_compute": len(missing_row_ids),
        "samples_needing_compute": len(missing_samples),
        "total_expected_rows": total_expected_rows,
    }

    if verbose:
        print(
            "[attempt4-study02] Noisy bbox expanded-row setup\n"
            "  config: {}\n"
            "  dataset_root: {}\n"
            "  total_images: {}\n"
            "  total_augmented_rows: {}\n"
            "  total_contexts: {}\n"
            "  total_aggregations: {}\n"
            "  num_jitter_boxes: {}\n"
            "  reused_rows: {}\n"
            "  missing_rows_to_compute: {}\n"
            "  samples_needing_compute: {}".format(
                config_path,
                dataset_root,
                len(all_samples),
                total_expected_rows,
                len(feature_layout["context_specs"]),
                len(feature_layout["aggregation_methods"]),
                len(jitter_specs),
                len(existing_rows),
                len(missing_row_ids),
                len(missing_samples),
            )
        )

    if dry_run:
        _write_feature_summary_checkpoint(feature_summary_json, existing_rows, summary_payload)
        return {
            "feature_table_csv": str(feature_table_csv),
            "feature_summary_json": str(feature_summary_json),
            "all_rows": existing_rows,
            "feature_layout": feature_layout,
        }

    depth_model = None
    if missing_samples:
        depth_config = dict(resolved_config.get("depth_model", {}))
        if verbose:
            print(
                "[attempt4-study02] Loading depth model\n"
                "  model_name: {}\n"
                "  device: {}\n"
                "  local_files_only: {}".format(
                    depth_config.get("model_name"),
                    depth_config.get("device"),
                    bool(depth_config.get("local_files_only", False)),
                )
            )
        model_name = depth_config.get("model_name")
        depth_model = load_depth_anything_v2(
            model_name=str(model_name) if model_name else None,
            device=depth_config.get("device"),
            local_files_only=bool(depth_config.get("local_files_only", False)),
        )

    progress_log_every_samples = max(1, int(study_config.get("progress_log_every_samples", 10)))
    all_rows = list(existing_rows)
    for sample_index, sample in enumerate(missing_samples, start=1):
        if verbose and (
            sample_index == 1
            or sample_index == len(missing_samples)
            or sample_index % progress_log_every_samples == 0
        ):
            print(
                "[attempt4-study02] Sample {}/{}\n"
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

        jitter_rows = _compute_jittered_feature_rows(
            sample=sample,
            loader=loader,
            depth_model=depth_model,
            context_specs=feature_layout["context_specs"],
            aggregation_methods=feature_layout["aggregation_methods"],
            jitter_specs=jitter_specs,
        )
        rows_written = 0
        for row in jitter_rows:
            if str(row["augmented_row_id"]) in existing_ids:
                continue
            _append_csv_row(row, feature_table_csv)
            all_rows.append(row)
            existing_ids.add(str(row["augmented_row_id"]))
            rows_written += 1

        if verbose and (
            sample_index == 1
            or sample_index == len(missing_samples)
            or sample_index % progress_log_every_samples == 0
        ):
            print(
                "[attempt4-study02]   saved {} jitter rows | rows now {}".format(
                    rows_written,
                    len(all_rows),
                )
            )
        _write_feature_summary_checkpoint(feature_summary_json, all_rows, summary_payload)

    all_rows = sorted(
        all_rows,
        key=lambda row: (
            float(row["true_distance_m"]),
            str(row["weather"]),
            str(row["time_of_day"]),
            str(row["source_image_path"]),
            str(row["jitter_name"]),
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


def _allocate_group_test_counts(
    stratum_to_group_ids: Dict[str, List[str]],
    holdout_fraction: float,
) -> Dict[str, int]:
    total_count = sum(len(group_ids) for group_ids in stratum_to_group_ids.values())
    target_test_count = int(round(total_count * holdout_fraction))

    floor_counts: Dict[str, int] = {}
    remainder_rows: List[Tuple[float, str]] = []
    capped_total = 0
    for stratum_key, group_ids in stratum_to_group_ids.items():
        exact = len(group_ids) * holdout_fraction
        base_count = int(math.floor(exact))
        base_count = min(base_count, max(0, len(group_ids) - 1))
        floor_counts[stratum_key] = base_count
        capped_total += base_count
        remainder_rows.append((exact - base_count, stratum_key))

    remaining = target_test_count - capped_total
    if remaining < 0:
        raise ValueError("Invalid grouped split allocation produced too many test groups.")

    for _, stratum_key in sorted(remainder_rows, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        current = floor_counts[stratum_key]
        max_allowed = max(0, len(stratum_to_group_ids[stratum_key]) - 1)
        if current < max_allowed:
            floor_counts[stratum_key] = current + 1
            remaining -= 1

    if remaining != 0:
        raise ValueError("Unable to allocate grouped holdout split with the available strata.")

    return floor_counts


def _collect_group_split_indices(
    test_split: Dict[str, Any],
    cv_folds_payload: Dict[str, Any],
) -> List[int]:
    indices = [int(index) for index in test_split.get("test_indices", [])]
    indices.extend(int(index) for index in test_split.get("dev_indices", []))
    for fold in cv_folds_payload.get("cv_folds", []):
        indices.extend(int(index) for index in fold.get("train_indices", []))
        indices.extend(int(index) for index in fold.get("val_indices", []))
    return indices


def _get_group_split_cache_invalidation_reason(
    rows: Sequence[Dict[str, Any]],
    test_split: Dict[str, Any],
    cv_folds_payload: Dict[str, Any],
    holdout_fraction: float,
    num_folds: int,
    seed: int,
) -> Optional[str]:
    if int(test_split.get("seed", -1)) != seed:
        return "saved test split seed differs from the current config"
    saved_holdout_fraction = float(test_split.get("holdout_fraction", -1.0))
    if not math.isclose(saved_holdout_fraction, holdout_fraction, rel_tol=0.0, abs_tol=1e-12):
        return "saved holdout fraction differs from the current config"
    if int(cv_folds_payload.get("seed", -1)) != seed:
        return "saved CV fold seed differs from the current config"
    if int(cv_folds_payload.get("num_folds", -1)) != num_folds:
        return "saved fold count differs from the current config"

    row_count = len(rows)
    all_indices = _collect_group_split_indices(test_split, cv_folds_payload)
    if any(index < 0 or index >= row_count for index in all_indices):
        return "saved split indices no longer fit the current feature-table row count"

    test_indices = [int(index) for index in test_split.get("test_indices", [])]
    dev_indices = [int(index) for index in test_split.get("dev_indices", [])]
    combined_indices = test_indices + dev_indices
    if len(combined_indices) != row_count or len(set(combined_indices)) != row_count:
        return "saved dev/test indices no longer partition the current feature-table rows"

    current_group_by_index = {
        str(index): str(row["source_group_id"]) for index, row in enumerate(rows)
    }
    saved_group_by_index = {
        str(index): str(group_id)
        for index, group_id in dict(test_split.get("group_by_index", {})).items()
    }
    if saved_group_by_index != current_group_by_index:
        return "saved row-to-group assignments no longer match the current feature-table order"

    dev_index_set = set(dev_indices)
    cv_folds = list(cv_folds_payload.get("cv_folds", []))
    if len(cv_folds) != num_folds:
        return "saved CV fold payload does not contain the expected number of folds"
    for fold in cv_folds:
        train_indices = {int(index) for index in fold.get("train_indices", [])}
        val_indices = {int(index) for index in fold.get("val_indices", [])}
        if train_indices & val_indices:
            return "saved CV fold contains overlapping train/validation indices"
        if train_indices | val_indices != dev_index_set:
            return "saved CV folds no longer cover the current dev partition"

    return None


def _make_grouped_holdout_and_cv_splits(
    rows: Sequence[Dict[str, Any]],
    holdout_fraction: float,
    num_folds: int,
    seed: int,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot split an empty row sequence.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    rng = random.Random(seed)
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    group_to_stratum: Dict[str, str] = {}
    row_group_ids: Dict[int, str] = {}

    for index, row in enumerate(rows):
        group_id = str(row["source_group_id"])
        stratum_key = "{}|{}|{}".format(
            make_distance_range_label(float(row["true_distance_m"])),
            str(row["weather"]),
            str(row["time_of_day"]),
        )
        group_to_indices[group_id].append(index)
        row_group_ids[index] = group_id
        existing_stratum = group_to_stratum.get(group_id)
        if existing_stratum is None:
            group_to_stratum[group_id] = stratum_key
        elif existing_stratum != stratum_key:
            raise ValueError("Group '{}' spans multiple strata.".format(group_id))

    stratum_to_group_ids: Dict[str, List[str]] = defaultdict(list)
    for group_id, stratum_key in group_to_stratum.items():
        stratum_to_group_ids[stratum_key].append(group_id)

    for group_ids in stratum_to_group_ids.values():
        rng.shuffle(group_ids)

    test_counts = _allocate_group_test_counts(stratum_to_group_ids, holdout_fraction)
    test_group_ids: List[str] = []
    dev_group_ids: List[str] = []
    dev_stratum_to_group_ids: Dict[str, List[str]] = {}

    for stratum_key, group_ids in stratum_to_group_ids.items():
        cutoff = test_counts[stratum_key]
        stratum_test_groups = group_ids[:cutoff]
        stratum_dev_groups = group_ids[cutoff:]
        test_group_ids.extend(stratum_test_groups)
        dev_group_ids.extend(stratum_dev_groups)
        dev_stratum_to_group_ids[stratum_key] = list(stratum_dev_groups)

    test_indices = sorted(
        index
        for group_id in test_group_ids
        for index in group_to_indices[group_id]
    )
    dev_indices = sorted(
        index
        for group_id in dev_group_ids
        for index in group_to_indices[group_id]
    )

    fold_group_buckets: List[List[str]] = [[] for _ in range(num_folds)]
    for stratum_key, group_ids in sorted(dev_stratum_to_group_ids.items()):
        shuffled_group_ids = list(group_ids)
        rng.shuffle(shuffled_group_ids)
        for offset, group_id in enumerate(shuffled_group_ids):
            fold_group_buckets[offset % num_folds].append(group_id)

    cv_folds: List[Dict[str, Any]] = []
    dev_group_set = set(dev_group_ids)
    for fold_index, val_group_ids in enumerate(fold_group_buckets):
        val_group_set = set(val_group_ids)
        train_group_ids = sorted(dev_group_set - val_group_set)
        val_indices = sorted(
            index for group_id in val_group_ids for index in group_to_indices[group_id]
        )
        train_indices = sorted(
            index for group_id in train_group_ids for index in group_to_indices[group_id]
        )
        cv_folds.append(
            {
                "fold_index": fold_index,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "val_group_ids": sorted(val_group_ids),
            }
        )

    return {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "num_folds": num_folds,
        "test_indices": test_indices,
        "dev_indices": dev_indices,
        "cv_folds": cv_folds,
        "group_by_index": {str(index): row_group_ids[index] for index in row_group_ids},
    }


def _load_or_create_group_splits(
    rows: Sequence[Dict[str, Any]],
    split_dir: Path,
    holdout_fraction: float,
    num_folds: int,
    seed: int,
) -> Dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    test_split_path = split_dir / "test_split.json"
    cv_folds_path = split_dir / "cv_folds.json"

    if test_split_path.exists() and cv_folds_path.exists():
        test_split = json.loads(test_split_path.read_text(encoding="utf-8"))
        cv_folds_payload = json.loads(cv_folds_path.read_text(encoding="utf-8"))
        invalidation_reason = _get_group_split_cache_invalidation_reason(
            rows=rows,
            test_split=test_split,
            cv_folds_payload=cv_folds_payload,
            holdout_fraction=holdout_fraction,
            num_folds=num_folds,
            seed=seed,
        )
        if invalidation_reason is None:
            return {
                "test_indices": test_split["test_indices"],
                "dev_indices": test_split["dev_indices"],
                "cv_folds": cv_folds_payload["cv_folds"],
                "seed": test_split["seed"],
                "holdout_fraction": test_split["holdout_fraction"],
                "num_folds": cv_folds_payload["num_folds"],
            }
        print(
            "[attempt4-study02] Rebuilding cached splits because {}".format(
                invalidation_reason
            )
        )

    split_data = _make_grouped_holdout_and_cv_splits(
        rows=rows,
        holdout_fraction=holdout_fraction,
        num_folds=num_folds,
        seed=seed,
    )
    test_split_payload = {
        "seed": split_data["seed"],
        "holdout_fraction": split_data["holdout_fraction"],
        "test_indices": split_data["test_indices"],
        "dev_indices": split_data["dev_indices"],
        "group_by_index": split_data["group_by_index"],
    }
    cv_folds_payload = {
        "seed": split_data["seed"],
        "num_folds": split_data["num_folds"],
        "cv_folds": split_data["cv_folds"],
    }
    _write_json(test_split_payload, test_split_path)
    _write_json(cv_folds_payload, cv_folds_path)
    return split_data


def _load_xgboost_regressor():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("xgboost is required for attempt4 Study 02: {}".format(exc)) from exc


def _fit_xgboost_estimator(params: Dict[str, Any]):
    estimator_cls = _load_xgboost_regressor()
    return estimator_cls(
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 4)),
        min_child_weight=float(params.get("min_child_weight", 2.0)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        random_state=int(params.get("random_state", 0)),
        n_jobs=int(params.get("n_jobs", 1)),
        objective="reg:squarederror",
        tree_method=str(params.get("tree_method", "hist")),
        verbosity=int(params.get("verbosity", 0)),
    )


def _fit_lower_xgb_model_and_predict(
    train_rows: Sequence[Dict[str, Any]],
    predict_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    xgb_params: Dict[str, Any],
) -> np.ndarray:
    if not predict_rows:
        return np.zeros(0, dtype=np.float64)
    estimator = _fit_xgboost_estimator(xgb_params)
    x_train = _build_feature_matrix(train_rows, lower_feature_names)
    y_train = np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64)
    estimator.fit(x_train, y_train)
    x_predict = _build_feature_matrix(predict_rows, lower_feature_names)
    return estimator.predict(x_predict)


def _cross_fit_lower_xgb_model(
    train_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    xgb_params: Dict[str, Any],
    num_folds: int,
    seed: int,
    verbose: bool = False,
    progress_label: str = "",
) -> Dict[str, Any]:
    if len(train_rows) < 2:
        raise ValueError("Need at least two rows to cross-fit the lower model.")

    inner_num_folds = max(2, min(int(num_folds), len(train_rows)))
    fold_payload = make_stratified_cv_folds(train_rows, num_folds=inner_num_folds, seed=seed)
    folds = fold_payload["cv_folds"]
    oof_predictions = np.zeros(len(train_rows), dtype=np.float64)

    for fold_index, fold in enumerate(folds, start=1):
        if verbose:
            print(
                "[attempt4-study02] Lower XGB cross-fit {}/{} | {} | train_rows={} | val_rows={}".format(
                    fold_index,
                    len(folds),
                    progress_label or "lower-model",
                    len(fold["train_indices"]),
                    len(fold["val_indices"]),
                )
            )
        inner_train_rows = _select_rows(train_rows, fold["train_indices"])
        inner_val_rows = _select_rows(train_rows, fold["val_indices"])
        fold_predictions = _fit_lower_xgb_model_and_predict(
            train_rows=inner_train_rows,
            predict_rows=inner_val_rows,
            lower_feature_names=lower_feature_names,
            xgb_params=xgb_params,
        )
        for local_index, prediction in zip(fold["val_indices"], fold_predictions):
            oof_predictions[int(local_index)] = float(prediction)

    full_estimator = _fit_xgboost_estimator(xgb_params)
    if verbose:
        print(
            "[attempt4-study02] Lower XGB refit on full split | {} | rows={}".format(
                progress_label or "lower-model",
                len(train_rows),
            )
        )
    full_estimator.fit(
        _build_feature_matrix(train_rows, lower_feature_names),
        np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64),
    )

    importance_rows = [
        {
            "model_name": "study02_jittered_small_xgb",
            "model_family": "xgboost",
            "term": feature_name,
            "importance_type": "feature_importance",
            "importance_value": float(importance_value),
        }
        for feature_name, importance_value in zip(
            lower_feature_names,
            getattr(full_estimator, "feature_importances_", np.zeros(len(lower_feature_names))),
        )
    ]
    return {
        "oof_predictions": oof_predictions,
        "full_estimator": full_estimator,
        "feature_importance_rows": importance_rows,
    }


def _prepare_outer_fold_xgb_caches(
    merged_rows: Sequence[Dict[str, Any]],
    cv_folds: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    lower_prediction_feature_name: str,
    lower_xgb_params: Dict[str, Any],
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
                    "[attempt4-study02] Preparing lower-model cache for outer fold {}/{} | train_rows={} | val_rows={}".format(
                        outer_fold_index,
                        len(cv_folds),
                        len(train_rows),
                        len(val_rows),
                    )
                )
            lower_cache = _cross_fit_lower_xgb_model(
                train_rows=train_rows,
                lower_feature_names=lower_feature_names,
                xgb_params=lower_xgb_params,
                num_folds=lower_inner_num_folds,
                seed=lower_inner_seed + int(fold["fold_index"]),
                verbose=verbose,
                progress_label="study02 outer_fold={}".format(outer_fold_index),
            )
            train_augmented = _attach_prediction_feature(
                train_rows,
                lower_prediction_feature_name,
                lower_cache["oof_predictions"],
            )
            val_lower_predictions = lower_cache["full_estimator"].predict(
                _build_feature_matrix(val_rows, lower_feature_names)
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


def _prepare_heldout_xgb_lower_cache(
    dev_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    lower_feature_names: Sequence[str],
    lower_prediction_feature_name: str,
    lower_xgb_params: Dict[str, Any],
    lower_inner_num_folds: int,
    lower_inner_seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    if verbose:
        print(
            "[attempt4-study02] Building shared lower-model feature for held-out test | dev_rows={} | test_rows={}".format(
                len(dev_rows),
                len(test_rows),
            )
        )
    lower_cache = _cross_fit_lower_xgb_model(
        train_rows=dev_rows,
        lower_feature_names=lower_feature_names,
        xgb_params=lower_xgb_params,
        num_folds=lower_inner_num_folds,
        seed=lower_inner_seed + 1000,
        verbose=verbose,
        progress_label="study02 heldout-test-prep",
    )
    dev_augmented = _attach_prediction_feature(
        dev_rows,
        lower_prediction_feature_name,
        lower_cache["oof_predictions"],
    )
    test_lower_predictions = lower_cache["full_estimator"].predict(
        _build_feature_matrix(test_rows, lower_feature_names)
    )
    test_augmented = _attach_prediction_feature(
        test_rows,
        lower_prediction_feature_name,
        test_lower_predictions,
    )
    return {
        "dev_lower_rows": dev_augmented,
        "test_lower_rows": test_augmented,
        "test_lower_predictions": test_lower_predictions,
        "feature_importance_rows": list(lower_cache["feature_importance_rows"]),
    }


def _candidate_params_by_name(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    candidate_rows = list(config.get("upper_xgboost", {}).get("candidate_configs", []))
    if not candidate_rows:
        raise ValueError("upper_xgboost.candidate_configs must not be empty.")
    output: Dict[str, Dict[str, Any]] = {}
    for row in candidate_rows:
        name = str(row["name"])
        if name in output:
            raise ValueError("Duplicate upper XGBoost candidate name '{}'.".format(name))
        params = dict(row)
        params.pop("name", None)
        output[name] = params
    return output


def _build_model_specs(
    config: Dict[str, Any],
    feature_layout: Dict[str, Any],
    lower_prediction_feature_name: str,
) -> List[Dict[str, Any]]:
    geometry_feature_names = list(feature_layout["geometry_feature_names"])
    raw_depth_feature_name = str(feature_layout["raw_depth_feature_name"])
    candidate_name_map = {
        str(key): [str(value) for value in values]
        for key, values in dict(config.get("upper_xgboost", {}).get("model_candidate_names", {})).items()
    }
    all_specs: Dict[str, Dict[str, Any]] = {
        "raw_depth_only_xgb": {
            "model_name": "raw_depth_only_xgb",
            "continuous_feature_names": [raw_depth_feature_name],
            "categorical_feature_names": [],
            "uses_lower_model": False,
        },
        "geometry_metadata_xgb": {
            "model_name": "geometry_metadata_xgb",
            "continuous_feature_names": geometry_feature_names,
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "raw_depth_geometry_metadata_xgb": {
            "model_name": "raw_depth_geometry_metadata_xgb",
            "continuous_feature_names": [raw_depth_feature_name] + geometry_feature_names,
            "categorical_feature_names": list(BASE_CATEGORICAL_FEATURE_NAMES),
            "uses_lower_model": False,
        },
        "small_xgb_depth_only_xgb": {
            "model_name": "small_xgb_depth_only_xgb",
            "continuous_feature_names": [lower_prediction_feature_name],
            "categorical_feature_names": [],
            "uses_lower_model": True,
        },
        "stacked_xgb_depth_geometry_metadata_xgb": {
            "model_name": "stacked_xgb_depth_geometry_metadata_xgb",
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


def _evaluate_upper_xgb_candidate(
    model_spec: Dict[str, Any],
    candidate_name: str,
    candidate_params: Dict[str, Any],
    outer_fold_caches: Dict[int, Dict[str, Any]],
    cv_folds: Sequence[Dict[str, Any]],
    dev_indices: Sequence[int],
    global_categorical_levels: Dict[str, Sequence[str]],
    verbose: bool,
) -> Dict[str, Any]:
    continuous_feature_names = list(model_spec["continuous_feature_names"])
    categorical_feature_names = list(model_spec["categorical_feature_names"])
    uses_lower_model = bool(model_spec["uses_lower_model"])
    feature_names = _build_tabular_feature_names(
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )

    fold_rows: List[Dict[str, Any]] = []
    dev_prediction_lookup: Dict[int, float] = {}
    dev_row_lookup: Dict[int, Dict[str, Any]] = {}
    for outer_fold_position, fold in enumerate(cv_folds, start=1):
        cache = outer_fold_caches[int(fold["fold_index"])]
        train_rows = cache["train_lower_rows"] if uses_lower_model else cache["train_base_rows"]
        val_rows = cache["val_lower_rows"] if uses_lower_model else cache["val_base_rows"]

        if verbose:
            print(
                "[attempt4-study02] Candidate evaluation | {} | {} | max_depth={} | outer_fold={}/{} | train_rows={} | val_rows={}".format(
                    model_spec["model_name"],
                    candidate_name,
                    candidate_params.get("max_depth"),
                    outer_fold_position,
                    len(cv_folds),
                    len(train_rows),
                    len(val_rows),
                )
            )

        estimator = _fit_xgboost_estimator(candidate_params)
        x_train = _build_tabular_matrix(
            train_rows,
            continuous_feature_names,
            categorical_feature_names,
            global_categorical_levels,
        )
        y_train = np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64)
        x_val = _build_tabular_matrix(
            val_rows,
            continuous_feature_names,
            categorical_feature_names,
            global_categorical_levels,
        )
        estimator.fit(x_train, y_train)
        val_predictions = estimator.predict(x_val)
        fold_eval = evaluate_predictions(val_rows, val_predictions)
        if verbose:
            print(
                "[attempt4-study02] Completed candidate fold | {} | {} | outer_fold={}/{} | fold_mae={:.4f}".format(
                    model_spec["model_name"],
                    candidate_name,
                    outer_fold_position,
                    len(cv_folds),
                    float(fold_eval["overall"]["mae"]),
                )
            )
        fold_rows.append(
            {
                "fold_index": int(fold["fold_index"]),
                "overall": fold_eval["overall"],
                "per_distance_range": fold_eval["per_distance_range"],
            }
        )
        for global_index, row, prediction in zip(cache["global_val_indices"], val_rows, val_predictions):
            dev_prediction_lookup[int(global_index)] = float(prediction)
            dev_row_lookup[int(global_index)] = dict(row)

    ordered_dev_rows = [dev_row_lookup[int(index)] for index in dev_indices]
    ordered_predictions = np.asarray(
        [dev_prediction_lookup[int(index)] for index in dev_indices],
        dtype=np.float64,
    )
    oof_eval = evaluate_predictions(ordered_dev_rows, ordered_predictions)
    prediction_rows = _prediction_rows(
        rows=ordered_dev_rows,
        predictions=ordered_predictions,
        model_name="{}__{}".format(model_spec["model_name"], candidate_name),
        split_name="cv_oof",
        model_family="xgboost",
    )
    return {
        "model_name": str(model_spec["model_name"]),
        "candidate_name": str(candidate_name),
        "candidate_params": dict(candidate_params),
        "feature_names": feature_names,
        "fold_metrics": fold_rows,
        "summary": _cv_summary_from_fold_rows(fold_rows),
        "oof_overall": oof_eval["overall"],
        "oof_per_distance_range": oof_eval["per_distance_range"],
        "oof_prediction_rows": prediction_rows,
        "oof_rows": ordered_dev_rows,
        "oof_predictions": ordered_predictions,
    }


def _candidate_metric_row(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(result["summary"])
    oof_overall = dict(result["oof_overall"])
    row = {
        "model_name": result["model_name"],
        "candidate_name": result["candidate_name"],
        "num_features": len(result["feature_names"]),
        "feature_names": " | ".join(result["feature_names"]),
        "oof_mae": float(oof_overall["mae"]),
        "oof_mean_relative_error": float(oof_overall["mean_relative_error"]),
        "oof_rmse": float(oof_overall["rmse"]),
        "oof_r2": float(oof_overall["r2"]),
        "oof_mean_signed_error": float(oof_overall["mean_signed_error"]),
    }
    for key, value in summary.items():
        row[str(key)] = value
    for key, value in result["candidate_params"].items():
        row["param_{}".format(key)] = value
    return row


def _select_best_candidate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No XGBoost candidate results were provided.")
    return sorted(
        results,
        key=lambda row: (
            float(row["oof_overall"]["mae"]),
            float(row["oof_overall"]["mean_relative_error"]),
            str(row["candidate_name"]),
        ),
    )[0]


def _fit_and_evaluate_selected_upper_xgb(
    model_spec: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    global_categorical_levels: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    continuous_feature_names = list(model_spec["continuous_feature_names"])
    categorical_feature_names = list(model_spec["categorical_feature_names"])
    estimator = _fit_xgboost_estimator(selected_candidate["candidate_params"])
    x_train = _build_tabular_matrix(
        train_rows,
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )
    y_train = np.asarray([float(row["true_distance_m"]) for row in train_rows], dtype=np.float64)
    x_test = _build_tabular_matrix(
        test_rows,
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )
    estimator.fit(x_train, y_train)
    test_predictions = estimator.predict(x_test)
    test_eval = evaluate_predictions(test_rows, test_predictions)
    feature_names = _build_tabular_feature_names(
        continuous_feature_names,
        categorical_feature_names,
        global_categorical_levels,
    )
    importance_rows = [
        {
            "model_name": selected_candidate["model_name"],
            "candidate_name": selected_candidate["candidate_name"],
            "model_family": "xgboost",
            "term": feature_name,
            "importance_type": "feature_importance",
            "importance_value": float(importance_value),
        }
        for feature_name, importance_value in zip(
            feature_names,
            getattr(estimator, "feature_importances_", np.zeros(len(feature_names))),
        )
    ]
    prediction_rows = _prediction_rows(
        rows=test_rows,
        predictions=test_predictions,
        model_name="{}__{}".format(selected_candidate["model_name"], selected_candidate["candidate_name"]),
        split_name="test",
        model_family="xgboost",
    )
    return {
        "metrics": test_eval,
        "prediction_rows": prediction_rows,
        "importance_rows": importance_rows,
        "predictions": test_predictions,
        "feature_names": feature_names,
    }


def _enrich_prediction_rows(
    prediction_rows: Sequence[Dict[str, Any]],
    source_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for prediction_row, source_row in zip(prediction_rows, source_rows):
        enriched = dict(prediction_row)
        enriched["source_image_path"] = str(source_row.get("source_image_path", ""))
        enriched["source_label_path"] = str(source_row.get("source_label_path", ""))
        enriched["jitter_name"] = str(source_row.get("jitter_name", ""))
        enriched["augmented_row_id"] = str(source_row.get("augmented_row_id", ""))
        output_rows.append(enriched)
    return output_rows


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
    selected_lower_model_name = "study02_jittered_small_xgb"
    selected_lower_feature_names = list(feature_layout["depth_feature_names"])
    model_specs = _build_model_specs(
        config=resolved_config,
        feature_layout=feature_layout,
        lower_prediction_feature_name=lower_prediction_feature_name,
    )
    needs_lower_model = any(bool(spec["uses_lower_model"]) for spec in model_specs)

    if dry_run:
        summary = {
            "study_name": str(resolved_config.get("study_name", "study02")),
            "num_rows": len(merged_rows),
            "num_lower_features": len(selected_lower_feature_names),
            "raw_depth_feature_name": str(feature_layout["raw_depth_feature_name"]),
            "model_names": [str(spec["model_name"]) for spec in model_specs],
            "feature_table_csv": feature_build["feature_table_csv"],
        }
        _write_json(summary, reports_dir / "summary.json")
        return summary

    split_config = dict(resolved_config.get("split", {}))
    split_data = _load_or_create_group_splits(
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
    lower_xgb_params = dict(lower_model_config.get("xgboost", {}))

    if verbose:
        print(
            "[attempt4-study02] Expanded jitter-row big-XGBoost study\n"
            "  config: {}\n"
            "  total_rows: {}\n"
            "  dev_rows: {}\n"
            "  test_rows: {}\n"
            "  unique_source_images: {}\n"
            "  selected_lower_model: {}\n"
            "  lower_feature_count: {}\n"
            "  requested_models: {}".format(
                config_path,
                len(merged_rows),
                len(dev_rows),
                len(test_rows),
                len({str(row['source_group_id']) for row in merged_rows}),
                selected_lower_model_name,
                len(selected_lower_feature_names),
                ", ".join(spec["model_name"] for spec in model_specs),
            )
        )

    outer_fold_caches = _prepare_outer_fold_xgb_caches(
        merged_rows=merged_rows,
        cv_folds=split_data["cv_folds"],
        lower_feature_names=selected_lower_feature_names,
        lower_prediction_feature_name=lower_prediction_feature_name,
        lower_xgb_params=lower_xgb_params,
        lower_inner_num_folds=lower_inner_num_folds,
        lower_inner_seed=lower_inner_seed,
        needs_lower_model=needs_lower_model,
        verbose=verbose,
    )

    upper_candidate_params = _candidate_params_by_name(resolved_config)
    cv_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study02")),
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
                "[attempt4-study02] CV evaluation {}/{} | {} | uses_lower_model={}".format(
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
                    "[attempt4-study02] Evaluating candidate {}/{} | {} | {}".format(
                        candidate_index,
                        len(model_spec["candidate_names"]),
                        model_spec["model_name"],
                        candidate_name,
                    )
                )
            result = _evaluate_upper_xgb_candidate(
                model_spec=model_spec,
                candidate_name=str(candidate_name),
                candidate_params=candidate_params,
                outer_fold_caches=outer_fold_caches,
                cv_folds=split_data["cv_folds"],
                dev_indices=split_data["dev_indices"],
                global_categorical_levels=global_categorical_levels,
                verbose=verbose,
            )
            result["oof_prediction_rows"] = _enrich_prediction_rows(
                result["oof_prediction_rows"],
                result["oof_rows"],
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
                "num_features": len(best_candidate["feature_names"]),
                "feature_names": " | ".join(best_candidate["feature_names"]),
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
                "[attempt4-study02] Selected best candidate | {} | {} | oof_mae={:.4f}".format(
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
            "[attempt4-study02] CV winner selected | {} | {} | oof_mae={:.4f}".format(
                winner_model_name,
                winner_candidate_name,
                float(ranked_models[0]["oof_mae"]),
            )
        )
    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["winner_candidate_name"] = winner_candidate_name
    cv_metrics["ranking"] = ranked_models
    _write_json(cv_metrics, reports_dir / "cv_metrics.json")
    _write_csv_rows(upper_candidate_rows, reports_dir / "upper_xgb_candidate_metrics.csv")
    _write_csv_rows(selected_model_rows, reports_dir / "upper_xgb_selected_models.csv")
    _write_csv_rows(cv_prediction_rows, reports_dir / "cv_predictions.csv")

    if needs_lower_model:
        heldout_lower_cache = _prepare_heldout_xgb_lower_cache(
            dev_rows=dev_rows,
            test_rows=test_rows,
            lower_feature_names=selected_lower_feature_names,
            lower_prediction_feature_name=lower_prediction_feature_name,
            lower_xgb_params=lower_xgb_params,
            lower_inner_num_folds=lower_inner_num_folds,
            lower_inner_seed=lower_inner_seed,
            verbose=verbose,
        )
    else:
        heldout_lower_cache = {
            "dev_lower_rows": dev_rows,
            "test_lower_rows": test_rows,
            "feature_importance_rows": [],
        }

    test_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study02")),
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
                "[attempt4-study02] Held-out test fit {}/{} | {} | candidate={}".format(
                    model_index,
                    len(model_specs),
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                )
            )
        train_rows = heldout_lower_cache["dev_lower_rows"] if model_spec["uses_lower_model"] else dev_rows
        eval_rows = heldout_lower_cache["test_lower_rows"] if model_spec["uses_lower_model"] else test_rows
        test_result = _fit_and_evaluate_selected_upper_xgb(
            model_spec=model_spec,
            selected_candidate=best_candidate,
            train_rows=train_rows,
            test_rows=eval_rows,
            global_categorical_levels=global_categorical_levels,
        )
        test_result["prediction_rows"] = _enrich_prediction_rows(
            test_result["prediction_rows"],
            eval_rows,
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
                "[attempt4-study02] Completed held-out test fit | {} | candidate={} | test_mae={:.4f}".format(
                    model_spec["model_name"],
                    best_candidate["candidate_name"],
                    float(test_result["metrics"]["overall"]["mae"]),
                )
            )

    _write_json(test_metrics, reports_dir / "test_metrics.json")
    _write_csv_rows(test_prediction_rows, reports_dir / "test_predictions.csv")
    _write_csv_rows(grouped_summary_rows, reports_dir / "grouped_error_summary.csv")
    _write_csv_rows(upper_feature_importance_rows, reports_dir / "upper_xgb_feature_importances.csv")
    _write_csv_rows(
        list(heldout_lower_cache.get("feature_importance_rows", [])),
        reports_dir / "lower_model_feature_importances.csv",
    )
    _write_json(
        {
            "selected_lower_model_name": selected_lower_model_name,
            "selected_lower_feature_names": list(selected_lower_feature_names),
            "num_selected_lower_features": len(selected_lower_feature_names),
            "xgboost_params": lower_xgb_params,
            "raw_depth_feature_name": str(feature_layout["raw_depth_feature_name"]),
        },
        reports_dir / "lower_model_summary.json",
    )

    winner_test_metrics = test_metrics["models"][winner_model_name]["overall"]
    summary_payload = {
        "study_name": str(resolved_config.get("study_name", "study02")),
        "num_rows": len(merged_rows),
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "num_unique_source_images": len({str(row["source_group_id"]) for row in merged_rows}),
        "num_jitter_rows_per_source": len(_resolve_jitter_specs(dict(resolved_config.get("study", {})))),
        "selected_lower_model_name": selected_lower_model_name,
        "selected_lower_feature_names": list(selected_lower_feature_names),
        "num_selected_lower_features": len(selected_lower_feature_names),
        "winner_model_name": winner_model_name,
        "winner_candidate_name": winner_candidate_name,
        "winner_cv_oof_mae": float(ranked_models[0]["oof_mae"]),
        "winner_cv_oof_mean_relative_error": float(ranked_models[0]["oof_mean_relative_error"]),
        "winner_test_mae": float(winner_test_metrics["mae"]),
        "winner_test_mean_relative_error": float(winner_test_metrics["mean_relative_error"]),
        "selected_upper_candidates_by_model": {
            str(model_name): str(result["candidate_name"])
            for model_name, result in selected_candidates.items()
        },
        "feature_table_csv": feature_build["feature_table_csv"],
        "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
        "test_metrics_json": str(reports_dir / "test_metrics.json"),
        "upper_xgb_candidate_metrics_csv": str(reports_dir / "upper_xgb_candidate_metrics.csv"),
    }
    _write_json(summary_payload, reports_dir / "summary.json")

    if verbose:
        print(
            "[attempt4-study02] Study complete\n"
            "  winner_model: {}\n"
            "  winner_candidate: {}\n"
            "  winner_test_mae: {:.4f}\n"
            "  summary_json: {}".format(
                winner_model_name,
                winner_candidate_name,
                float(winner_test_metrics["mae"]),
                reports_dir / "summary.json",
            )
        )
    return summary_payload


def main() -> None:
    args = parse_args()
    run_study(
        config_path=args.config,
        dry_run=bool(args.dry_run),
        verbose=True,
    )


if __name__ == "__main__":
    main()
