"""
Study 06: all-images selected-subset random forest.

This study mirrors the core Study 04 idea, but changes two things:
- use every available dataset image instead of a 30-per-stratum sample
- keep only a selected subset of contexts and aggregations from the config

The script reuses any cached rows from earlier studies, computes only the
missing rows for the configured subset, then trains and evaluates random-forest
models on the fused feature table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.depth_representation_features import (
    DEPTH_SCORE_FIELD_NAMES,
    DepthRepresentationRecord,
    SelectedSampleRecord,
    build_selected_sample_records,
    read_depth_representation_records_csv,
    summarize_depth_representation_records,
    write_depth_representation_records_csv,
    write_selected_sample_records_csv,
)
from pipeline.calibration.depth_representation_study import (
    _pearson_correlation,
    _spearman_correlation,
    build_method_id,
)
from pipeline.calibration.evaluation import evaluate_predictions
from pipeline.data.loader import DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import (
    SUPPORTED_AGGREGATION_METHODS,
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Study 06 all-images selected-subset random forest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 06 YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the planned workload without computing missing depth rows or training models.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_score_fields(study_config: Dict[str, Any]) -> List[str]:
    requested = list(study_config.get("score_fields", DEPTH_SCORE_FIELD_NAMES))
    invalid = [name for name in requested if name not in DEPTH_SCORE_FIELD_NAMES]
    if invalid:
        raise ValueError(
            "Unsupported score_fields {}. Expected a subset of {}.".format(
                invalid,
                DEPTH_SCORE_FIELD_NAMES,
            )
        )
    return requested


def _normalize_path(path: str) -> str:
    return str(Path(path).resolve())


def _record_key(image_path: str, context_window: str, aggregation_method: str) -> Tuple[str, str, str]:
    return (_normalize_path(image_path), str(context_window), str(aggregation_method))


def _sample_stratum_key(sample: Any) -> Tuple[float, str, str]:
    return (float(sample.true_distance_m), str(sample.weather), str(sample.time_of_day))


def _build_all_selected_samples(samples: Sequence[Any]) -> List[Tuple[Any, int]]:
    grouped: Dict[Tuple[float, str, str], List[Any]] = defaultdict(list)
    for sample in samples:
        grouped[_sample_stratum_key(sample)].append(sample)

    selected: List[Tuple[Any, int]] = []
    for stratum_key in sorted(grouped):
        candidates = sorted(grouped[stratum_key], key=lambda item: str(item.image_path))
        for selection_rank, candidate in enumerate(candidates, start=1):
            selected.append((candidate, selection_rank))

    selected.sort(
        key=lambda item: (
            float(item[0].true_distance_m),
            str(item[0].weather),
            str(item[0].time_of_day),
            str(item[0].image_path),
        )
    )
    return selected


def _load_existing_records(
    source_paths: Sequence[Path],
    selected_image_paths: Iterable[str],
    desired_contexts: Sequence[str],
    desired_aggregations: Sequence[str],
) -> Tuple[Dict[Tuple[str, str, str], DepthRepresentationRecord], Dict[str, int]]:
    selected_set = {_normalize_path(path) for path in selected_image_paths}
    desired_context_set = set(desired_contexts)
    desired_aggregation_set = set(desired_aggregations)
    records_by_key: Dict[Tuple[str, str, str], DepthRepresentationRecord] = {}
    retained_counts: Counter[str] = Counter()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        source_records = read_depth_representation_records_csv(source_path)
        for record in source_records:
            image_key = _normalize_path(record.image_path)
            if image_key not in selected_set:
                continue
            if record.context_window not in desired_context_set:
                continue
            if record.aggregation_method not in desired_aggregation_set:
                continue
            key = (image_key, record.context_window, record.aggregation_method)
            if key in records_by_key:
                continue
            records_by_key[key] = record
            retained_counts[str(source_path)] += 1
    return records_by_key, dict(retained_counts)


def _build_record(
    selected_sample: SelectedSampleRecord,
    context_name: str,
    context_scale: Optional[float],
    aggregation_method: str,
    crop_bbox: Sequence[int],
    local_bbox: Sequence[int],
    representation: Dict[str, Any],
) -> DepthRepresentationRecord:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    local_x1, local_y1, local_x2, local_y2 = local_bbox
    focus_x1, focus_y1, focus_x2, focus_y2 = representation["focus_bbox"]
    ring_x1, ring_y1, ring_x2, ring_y2 = representation["surrounding_bbox"]
    return DepthRepresentationRecord(
        image_path=str(selected_sample.image_path),
        label_path=str(selected_sample.label_path),
        stratum_key=str(selected_sample.stratum_key),
        selection_rank_within_stratum=int(selected_sample.selection_rank_within_stratum),
        true_distance_m=float(selected_sample.true_distance_m),
        weather=str(selected_sample.weather),
        time_of_day=str(selected_sample.time_of_day),
        context_window=str(context_name),
        context_scale="full_image" if context_scale is None else str(context_scale),
        aggregation_method=str(aggregation_method),
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
        object_depth=round(float(representation["object_depth"]), 8),
        object_depth_percentile_5_95=round(float(representation["object_depth_percentile_5_95"]), 8),
        surrounding_ring_depth=round(float(representation["surrounding_ring_depth"]), 8),
        object_minus_ring=round(float(representation["object_minus_ring"]), 8),
        object_minus_ring_normalized=round(float(representation["object_minus_ring_normalized"]), 8),
        robust_global_range=round(float(representation["robust_global_range"]), 8),
        global_percentile_5=round(float(representation["global_percentile_5"]), 8),
        global_percentile_95=round(float(representation["global_percentile_95"]), 8),
    )


def _write_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv_rows(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_representation_records_csv(
    records: Sequence[DepthRepresentationRecord],
    output_path: Path,
) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DepthRepresentationRecord.__dataclass_fields__.keys())
        if not file_exists or output_path.stat().st_size == 0:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
        handle.flush()
        os.fsync(handle.fileno())


def _write_feature_summary_checkpoint(
    output_path: Path,
    all_records: Sequence[DepthRepresentationRecord],
    selected_sample_records: Sequence[SelectedSampleRecord],
    summary_payload: Dict[str, Any],
) -> None:
    summary = summarize_depth_representation_records(all_records, selected_sample_records)
    summary.update(summary_payload)
    summary["representation_records_csv"] = str(output_path.parent / "representation_records.csv")
    _write_json(summary, output_path)


def _build_feature_cache(
    resolved_config: Dict[str, Any],
    config_path: Path,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, Any]:
    output_root = Path(resolved_config["output_root"])
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    features_csv_path = features_dir / "representation_records.csv"
    features_summary_path = features_dir / "summary.json"

    study_config = resolved_config.get("study", {})
    source_paths = []
    if features_csv_path.exists():
        source_paths.append(features_csv_path)
    source_paths.extend(
        Path(path) for path in resolved_config.get("existing_feature_sources", {}).get("representation_records_csvs", [])
    )
    dataset_root = Path(resolved_config["dataset_root"])
    strict = bool(resolved_config.get("strict", True))

    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=strict)
    all_samples = list(loader.iter_samples())
    selected_samples = _build_all_selected_samples(all_samples)
    selected_sample_records = build_selected_sample_records(selected_samples)
    selected_image_paths = [row.image_path for row in selected_sample_records]

    context_specs = resolve_context_window_specs(study_config.get("context_windows"))
    context_order = [spec.name for spec in context_specs]
    aggregation_methods = list(study_config.get("aggregation_methods", ["bbox_midpoint"]))
    score_fields = _resolve_score_fields(study_config)
    for aggregation_method in aggregation_methods:
        if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                "Unsupported aggregation_method '{}'. Expected one of {}.".format(
                    aggregation_method,
                    SUPPORTED_AGGREGATION_METHODS,
                )
            )

    existing_records_by_key, retained_counts = _load_existing_records(
        source_paths=source_paths,
        selected_image_paths=selected_image_paths,
        desired_contexts=context_order,
        desired_aggregations=aggregation_methods,
    )

    total_expected_rows = len(selected_sample_records) * len(context_order) * len(aggregation_methods)
    missing_keys: List[Tuple[str, str, str]] = []
    for selected_sample in selected_sample_records:
        for context_name in context_order:
            for aggregation_method in aggregation_methods:
                key = _record_key(selected_sample.image_path, context_name, aggregation_method)
                if key not in existing_records_by_key:
                    missing_keys.append(key)

    selected_samples_output_path = write_selected_sample_records_csv(
        selected_sample_records,
        features_dir / "selected_samples.csv",
    )

    summary_payload: Dict[str, Any] = {
        "study_name": resolved_config.get("study_name", output_root.name),
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "num_total_images": len(all_samples),
        "num_strata": len({_sample_stratum_key(sample) for sample in all_samples}),
        "num_selected_samples": len(selected_sample_records),
        "selected_samples_csv": str(selected_samples_output_path),
        "desired_contexts": list(context_order),
        "desired_aggregations": list(aggregation_methods),
        "desired_score_fields": list(score_fields),
        "reused_rows": len(existing_records_by_key),
        "missing_rows_to_compute": len(missing_keys),
        "total_expected_rows": total_expected_rows,
        "source_representation_records_csvs": [str(path) for path in source_paths],
        "source_rows_reused_by_file": retained_counts,
    }

    if verbose:
        print(
            "[study-06] All-images selected-subset setup\n"
            "  config: {}\n"
            "  dataset_root: {}\n"
            "  total_images: {}\n"
            "  num_strata: {}\n"
            "  selected_images: {}\n"
            "  desired_contexts: {}\n"
            "  desired_aggregations: {}\n"
            "  desired_score_fields: {}\n"
            "  reusable_rows: {}\n"
            "  missing_rows_to_compute: {}\n"
            "  total_expected_rows: {}".format(
                config_path,
                dataset_root,
                len(all_samples),
                len({_sample_stratum_key(sample) for sample in all_samples}),
                len(selected_sample_records),
                ", ".join(context_order),
                ", ".join(aggregation_methods),
                ", ".join(score_fields),
                len(existing_records_by_key),
                len(missing_keys),
                total_expected_rows,
            )
        )

    if dry_run:
        summary_payload["dry_run"] = True
        _write_json(summary_payload, features_summary_path)
        return summary_payload

    all_records: List[DepthRepresentationRecord] = list(existing_records_by_key.values())
    if not features_csv_path.exists():
        existing_sorted = sorted(
            all_records,
            key=lambda record: (
                float(record.true_distance_m),
                str(record.weather),
                str(record.time_of_day),
                str(record.image_path),
                str(record.context_window),
                str(record.aggregation_method),
            ),
        )
        write_depth_representation_records_csv(existing_sorted, features_csv_path)
        _write_feature_summary_checkpoint(
            output_path=features_summary_path,
            all_records=existing_sorted,
            selected_sample_records=selected_sample_records,
            summary_payload=summary_payload,
        )
    else:
        _write_feature_summary_checkpoint(
            output_path=features_summary_path,
            all_records=all_records,
            selected_sample_records=selected_sample_records,
            summary_payload=summary_payload,
        )

    if missing_keys:
        sample_lookup = {_normalize_path(str(sample.image_path)): sample for sample in all_samples}
        depth_config = resolved_config.get("depth_model", {})
        if verbose:
            print(
                "[study-06] Loading depth model\n"
                "  model_name: {}\n"
                "  device: {}\n"
                "  local_files_only: {}".format(
                    depth_config.get("model_name"),
                    depth_config.get("device"),
                    depth_config.get("local_files_only", False),
                )
            )
        depth_model = load_depth_anything_v2(
            model_name=depth_config.get("model_name"),
            device=depth_config.get("device"),
            local_files_only=bool(depth_config.get("local_files_only", False)),
        )

        progress_log_every_samples = max(1, int(study_config.get("progress_log_every_samples", 25)))
        progress_log_every_contexts = max(1, int(study_config.get("progress_log_every_contexts", 1)))

        for sample_index, selected_sample in enumerate(selected_sample_records, start=1):
            new_records_for_sample: List[DepthRepresentationRecord] = []
            missing_context_specs = []
            for context_spec in context_specs:
                missing_aggs = [
                    aggregation_method
                    for aggregation_method in aggregation_methods
                    if _record_key(selected_sample.image_path, context_spec.name, aggregation_method)
                    not in existing_records_by_key
                ]
                if missing_aggs:
                    missing_context_specs.append((context_spec, missing_aggs))

            if not missing_context_specs:
                continue

            if verbose and (
                sample_index == 1
                or sample_index == len(selected_sample_records)
                or sample_index % progress_log_every_samples == 0
            ):
                print(
                    "[study-06] Sample {}/{}\n"
                    "  distance: {} m\n"
                    "  weather: {}\n"
                    "  time_of_day: {}\n"
                    "  image: {}\n"
                    "  missing_contexts: {}".format(
                        sample_index,
                        len(selected_sample_records),
                        selected_sample.true_distance_m,
                        selected_sample.weather,
                        selected_sample.time_of_day,
                        selected_sample.image_path,
                        ", ".join(spec.name for spec, _ in missing_context_specs),
                    )
                )

            sample = sample_lookup[_normalize_path(selected_sample.image_path)]
            image = DroneDatasetLoader.load_image(sample)
            for context_index, (context_spec, missing_aggs) in enumerate(missing_context_specs, start=1):
                crop_data = extract_context_crop(
                    image=image,
                    bbox=sample.annotation.bbox,
                    context_spec=context_spec,
                )
                depth_map = estimate_relative_depth(crop_data["image"], depth_model)
                for aggregation_method in missing_aggs:
                    representation = compute_depth_representation(
                        depth_map=depth_map,
                        bbox=crop_data["local_bbox"],
                        aggregation_method=aggregation_method,
                        inner_bbox_scale=float(study_config.get("inner_bbox_scale", 0.5)),
                        surrounding_bbox_scale=float(study_config.get("surrounding_bbox_scale", 1.8)),
                    )
                    record = _build_record(
                        selected_sample=selected_sample,
                        context_name=context_spec.name,
                        context_scale=context_spec.scale,
                        aggregation_method=aggregation_method,
                        crop_bbox=crop_data["crop_bbox"],
                        local_bbox=crop_data["local_bbox"],
                        representation=representation,
                    )
                    key = _record_key(record.image_path, record.context_window, record.aggregation_method)
                    existing_records_by_key[key] = record
                    all_records.append(record)
                    new_records_for_sample.append(record)
                if verbose and (
                    context_index == 1
                    or context_index == len(missing_context_specs)
                    or context_index % progress_log_every_contexts == 0
                ):
                    print(
                        "[study-06]   context {}/{} | {} | rows now {}".format(
                            context_index,
                            len(missing_context_specs),
                            context_spec.name,
                            len(all_records),
                        )
                    )
            if new_records_for_sample:
                _append_representation_records_csv(new_records_for_sample, features_csv_path)
                _write_feature_summary_checkpoint(
                    output_path=features_summary_path,
                    all_records=all_records,
                    selected_sample_records=selected_sample_records,
                    summary_payload=summary_payload,
                )

    all_records = sorted(
        all_records,
        key=lambda record: (
            float(record.true_distance_m),
            str(record.weather),
            str(record.time_of_day),
            str(record.image_path),
            str(record.context_window),
            str(record.aggregation_method),
        ),
    )
    features_csv_path = write_depth_representation_records_csv(
        all_records,
        features_csv_path,
    )
    summary = summarize_depth_representation_records(all_records, selected_sample_records)
    summary.update(summary_payload)
    summary["representation_records_csv"] = str(features_csv_path)
    _write_json(summary, features_summary_path)
    return summary


def _standardize_train_matrix(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    return (x_train - means) / scales, means, scales


def _fit_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    x_train_standardized, means, scales = _standardize_train_matrix(x_train)
    design = np.column_stack([np.ones(x_train_standardized.shape[0], dtype=np.float64), x_train_standardized])
    coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    intercept_std = float(coefficients[0])
    coef_std = np.asarray(coefficients[1:], dtype=np.float64)
    coef_original = coef_std / scales
    intercept_original = intercept_std - float(np.sum(coef_std * means / scales))
    return {
        "intercept": float(intercept_original),
        "coefficients": coef_original.astype(np.float64),
    }


def _predict_linear_regression(x_values: np.ndarray, fit_result: Dict[str, Any]) -> np.ndarray:
    return float(fit_result["intercept"]) + np.asarray(x_values, dtype=np.float64) @ np.asarray(
        fit_result["coefficients"], dtype=np.float64
    )


def _pivot_records_to_image_rows(
    records: Sequence[DepthRepresentationRecord],
    score_fields: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows_by_image: Dict[str, Dict[str, Any]] = {}
    feature_names: List[str] = []
    for record in records:
        image_key = str(record.image_path)
        row = rows_by_image.setdefault(
            image_key,
            {
                "image_path": str(record.image_path),
                "label_path": str(record.label_path),
                "stratum_key": str(record.stratum_key),
                "selection_rank_within_stratum": int(record.selection_rank_within_stratum),
                "true_distance_m": float(record.true_distance_m),
                "weather": str(record.weather),
                "time_of_day": str(record.time_of_day),
            },
        )
        for score_field in score_fields:
            method_id = build_method_id(record.context_window, record.aggregation_method, score_field)
            row[method_id] = float(getattr(record, score_field))
            if method_id not in feature_names:
                feature_names.append(method_id)

    rows = sorted(
        rows_by_image.values(),
        key=lambda row: (
            float(row["true_distance_m"]),
            str(row["weather"]),
            str(row["time_of_day"]),
            str(row["image_path"]),
        ),
    )
    for row in rows:
        for feature_name in feature_names:
            if feature_name not in row:
                raise ValueError("Missing feature '{}' in fused row.".format(feature_name))
    return rows, feature_names


def _make_balanced_folds(
    rows: Sequence[Dict[str, Any]],
    num_folds: int,
    seed: int,
) -> List[np.ndarray]:
    grouped_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(str(row["stratum_key"]), []).append(index)
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for stratum_key in sorted(grouped_indices):
        indices = sorted(
            grouped_indices[stratum_key],
            key=lambda idx: (
                int(rows[idx]["selection_rank_within_stratum"]),
                str(rows[idx]["image_path"]),
            ),
        )
        permuted = list(indices)
        rng.shuffle(permuted)
        for rank, row_index in enumerate(permuted):
            folds[rank % num_folds].append(row_index)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def _evaluate_single_feature(
    rows: Sequence[Dict[str, Any]],
    feature_name: str,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    x_values = np.asarray([float(row[feature_name]) for row in rows], dtype=np.float64)
    predictions = np.zeros(true_values.shape[0], dtype=np.float64)
    for fold_indices in folds:
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[fold_indices] = False
        fit_result = _fit_linear_regression(
            x_train=x_values[train_mask].reshape(-1, 1),
            y_train=true_values[train_mask],
        )
        predictions[fold_indices] = _predict_linear_regression(x_values[fold_indices].reshape(-1, 1), fit_result)
    evaluation = evaluate_predictions(rows, predictions)
    return {
        "method_id": feature_name,
        "count": evaluation["overall"]["count"],
        "pearson_correlation": _pearson_correlation(x_values, true_values),
        "spearman_correlation": _spearman_correlation(x_values, true_values),
        "cv_mae": evaluation["overall"]["mae"],
        "cv_median_absolute_error": evaluation["overall"]["median_absolute_error"],
        "cv_mean_relative_error": evaluation["overall"]["mean_relative_error"],
        "cv_median_relative_error": evaluation["overall"]["median_relative_error"],
        "cv_rmse": evaluation["overall"]["rmse"],
        "cv_r2": evaluation["overall"]["r2"],
        "cv_mean_signed_error": evaluation["overall"]["mean_signed_error"],
        "cv_median_signed_error": evaluation["overall"]["median_signed_error"],
        "cv_p90_absolute_error": evaluation["overall"]["p90_absolute_error"],
        "cv_p95_absolute_error": evaluation["overall"]["p95_absolute_error"],
        "cv_max_absolute_error": evaluation["overall"]["max_absolute_error"],
        "cv_within_5m_rate": evaluation["overall"]["within_5m_rate"],
        "cv_within_10m_rate": evaluation["overall"]["within_10m_rate"],
        "cv_within_20m_rate": evaluation["overall"]["within_20m_rate"],
        "cv_overprediction_rate": evaluation["overall"]["overprediction_rate"],
        "cv_underprediction_rate": evaluation["overall"]["underprediction_rate"],
    }


def _load_random_forest_regressor():
    try:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("scikit-learn is required for Study 06 random forest: {}".format(exc)) from exc


def _fit_random_forest_estimator(params: Dict[str, Any]):
    estimator_cls = _load_random_forest_regressor()
    return estimator_cls(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        max_features=params.get("max_features", "sqrt"),
        random_state=int(params.get("random_state", 0)),
        n_jobs=-1,
    )


def _evaluate_random_forest_model(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    folds: Sequence[np.ndarray],
    model_name: str,
    params: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    x_values = np.asarray(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=np.float64,
    )
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    predictions = np.zeros(true_values.shape[0], dtype=np.float64)
    for fold_index, fold_indices in enumerate(folds, start=1):
        if verbose:
            print(
                "[study-06] RF fold {}/{} | {} | {} feature(s)".format(
                    fold_index,
                    len(folds),
                    model_name,
                    len(feature_names),
                )
            )
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[fold_indices] = False
        estimator = _fit_random_forest_estimator(params)
        estimator.fit(x_values[train_mask], true_values[train_mask])
        predictions[fold_indices] = estimator.predict(x_values[fold_indices])

    estimator = _fit_random_forest_estimator(params)
    estimator.fit(x_values, true_values)
    evaluation = evaluate_predictions(rows, predictions)
    importance_rows = [
        {
            "model_name": model_name,
            "model_family": "random_forest",
            "term": feature_name,
            "importance_type": "feature_importance",
            "importance_value": float(importance_value),
        }
        for feature_name, importance_value in zip(feature_names, estimator.feature_importances_)
    ]
    prediction_rows = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        signed_error = float(prediction - true_distance)
        prediction_rows.append(
            {
                "model_name": model_name,
                "model_family": "random_forest",
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "stratum_key": row["stratum_key"],
                "selection_rank_within_stratum": row["selection_rank_within_stratum"],
                "weather": row["weather"],
                "time_of_day": row["time_of_day"],
                "true_distance_m": true_distance,
                "predicted_distance_m": float(prediction),
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
            }
        )
    return {
        "model_name": model_name,
        "model_family": "random_forest",
        "feature_names": list(feature_names),
        "predictions": predictions,
        "prediction_rows": prediction_rows,
        "metrics": evaluation["overall"],
        "importance_rows": importance_rows,
    }


def _run_modeling(
    resolved_config: Dict[str, Any],
    records: Sequence[DepthRepresentationRecord],
    config_path: Path,
    verbose: bool,
) -> Dict[str, Any]:
    output_root = Path(resolved_config["output_root"])
    fused_features_dir = output_root / "fused_features"
    reports_dir = output_root / "reports"
    fused_features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    score_fields = _resolve_score_fields(resolved_config.get("study", {}))
    image_rows, available_feature_names = _pivot_records_to_image_rows(records, score_fields=score_fields)
    fused_features_path = fused_features_dir / "depth_only_feature_table.csv"
    _write_csv_rows(image_rows, fused_features_path)

    modeling_config = resolved_config.get("modeling", {})
    num_folds = int(modeling_config.get("num_folds", 5))
    cv_seed = int(modeling_config.get("cv_seed", 0))
    folds = _make_balanced_folds(image_rows, num_folds=num_folds, seed=cv_seed)

    if verbose:
        print(
            "[study-06] Selected-subset random-forest modeling\n"
            "  config: {}\n"
            "  fused_features_csv: {}\n"
            "  num_image_rows: {}\n"
            "  num_available_features: {}\n"
            "  score_fields: {}\n"
            "  num_folds: {}".format(
                config_path,
                fused_features_path,
                len(image_rows),
                len(available_feature_names),
                ", ".join(score_fields),
                num_folds,
            )
        )

    single_feature_rows: List[Dict[str, Any]] = []
    progress_every = max(1, int(resolved_config.get("study", {}).get("progress_log_every_single_features", 10)))
    for feature_index, feature_name in enumerate(available_feature_names, start=1):
        if verbose and (feature_index == 1 or feature_index == len(available_feature_names) or feature_index % progress_every == 0):
            print(
                "[study-06] Single-feature baseline {}/{}\n"
                "  feature: {}".format(
                    feature_index,
                    len(available_feature_names),
                    feature_name,
                )
            )
        result = _evaluate_single_feature(image_rows, feature_name, folds)
        single_feature_rows.append(result)

    single_feature_rows = sorted(
        single_feature_rows,
        key=lambda row: (
            float(row["cv_mae"]),
            float(row["cv_mean_relative_error"]),
            -float(row["spearman_correlation"]),
            str(row["method_id"]),
        ),
    )
    for rank_index, row in enumerate(single_feature_rows, start=1):
        row["rank"] = rank_index
        context_window, aggregation_method, score_field = str(row["method_id"]).split("__", 2)
        row["context_window"] = context_window
        row["aggregation_method"] = aggregation_method
        row["score_field"] = score_field
    _write_csv_rows(single_feature_rows, reports_dir / "single_feature_cv_metrics.csv")

    rf_params = dict(modeling_config.get("random_forest", {}))
    model_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    candidate_results: List[Dict[str, Any]] = []
    for top_k in [int(value) for value in modeling_config.get("top_feature_pool_sizes", [12, 24])]:
        feature_names = [row["method_id"] for row in single_feature_rows[:top_k]]
        result = _evaluate_random_forest_model(
            rows=image_rows,
            feature_names=feature_names,
            folds=folds,
            model_name="random_forest_top_{}".format(top_k),
            params=rf_params,
            verbose=verbose,
        )
        candidate_results.append(result)
        metrics = result["metrics"]
        model_rows.append(
            {
                "model_name": result["model_name"],
                "model_family": result["model_family"],
                "num_features": len(feature_names),
                "feature_names": " | ".join(feature_names),
                "cv_mae": float(metrics["mae"]),
                "cv_median_absolute_error": float(metrics["median_absolute_error"]),
                "cv_mean_relative_error": float(metrics["mean_relative_error"]),
                "cv_median_relative_error": float(metrics["median_relative_error"]),
                "cv_rmse": float(metrics["rmse"]),
                "cv_r2": float(metrics["r2"]),
                "cv_mean_signed_error": float(metrics["mean_signed_error"]),
                "cv_median_signed_error": float(metrics["median_signed_error"]),
                "cv_p90_absolute_error": float(metrics["p90_absolute_error"]),
                "cv_p95_absolute_error": float(metrics["p95_absolute_error"]),
                "cv_max_absolute_error": float(metrics["max_absolute_error"]),
                "cv_within_5m_rate": float(metrics["within_5m_rate"]),
                "cv_within_10m_rate": float(metrics["within_10m_rate"]),
                "cv_within_20m_rate": float(metrics["within_20m_rate"]),
                "cv_overprediction_rate": float(metrics["overprediction_rate"]),
                "cv_underprediction_rate": float(metrics["underprediction_rate"]),
            }
        )
        importance_rows.extend(result["importance_rows"])
        prediction_rows.extend(result["prediction_rows"])

    model_rows = sorted(model_rows, key=lambda row: (float(row["cv_mae"]), float(row["cv_mean_relative_error"])))
    best_model = model_rows[0]
    for rank_index, row in enumerate(model_rows, start=1):
        row["rank"] = rank_index

    _write_csv_rows(model_rows, reports_dir / "subset_random_forest_metrics.csv")
    _write_csv_rows(importance_rows, reports_dir / "subset_random_forest_feature_importances.csv")
    _write_csv_rows(prediction_rows, reports_dir / "subset_random_forest_predictions.csv")

    summary = {
        "study_name": resolved_config.get("study_name", output_root.name),
        "config_path": str(config_path),
        "num_image_rows": len(image_rows),
        "num_available_features": len(available_feature_names),
        "score_fields": list(score_fields),
        "num_folds": num_folds,
        "best_single_feature": single_feature_rows[0]["method_id"],
        "best_single_feature_cv_mae": float(single_feature_rows[0]["cv_mae"]),
        "best_random_forest_model": best_model["model_name"],
        "best_random_forest_cv_mae": float(best_model["cv_mae"]),
        "best_random_forest_num_features": int(best_model["num_features"]),
        "fused_features_csv": str(fused_features_path),
        "single_feature_metrics_csv": str(reports_dir / "single_feature_cv_metrics.csv"),
        "random_forest_metrics_csv": str(reports_dir / "subset_random_forest_metrics.csv"),
        "feature_importances_csv": str(reports_dir / "subset_random_forest_feature_importances.csv"),
        "predictions_csv": str(reports_dir / "subset_random_forest_predictions.csv"),
    }
    _write_json(summary, reports_dir / "summary.json")
    return summary


def run_study_06(
    config_path: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = load_config(config_path)
    resolved_path = Path(config_path or DEFAULT_CONFIG_PATH)
    feature_summary = _build_feature_cache(
        resolved_config=resolved_config,
        config_path=resolved_path,
        dry_run=dry_run,
        verbose=verbose,
    )
    if dry_run:
        return feature_summary

    features_csv_path = Path(resolved_config["output_root"]) / "features" / "representation_records.csv"
    records = read_depth_representation_records_csv(features_csv_path)

    if not bool(resolved_config.get("modeling", {}).get("enabled", True)):
        return feature_summary

    modeling_summary = _run_modeling(
        resolved_config=resolved_config,
        records=records,
        config_path=resolved_path,
        verbose=verbose,
    )
    return {
        "feature_summary": feature_summary,
        "modeling_summary": modeling_summary,
    }


def main() -> None:
    args = parse_args()
    run_study_06(config_path=args.config, dry_run=bool(args.dry_run), verbose=True)


if __name__ == "__main__":
    main()
