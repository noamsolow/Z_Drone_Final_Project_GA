"""
Incrementally extend the Study 03 feature cache with any missing zoom windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
#""fdere
from pipeline.calibration.depth_representation_features import (
    DepthRepresentationRecord,
    SelectedSampleRecord,
    read_depth_representation_records_csv,
    summarize_depth_representation_records,
    write_depth_representation_records_csv,
    write_selected_sample_records_csv,
)
from pipeline.data.loader import DroneDatasetLoader
from pipeline.depth.depth_model import estimate_relative_depth, load_depth_anything_v2
from pipeline.depth.representation import (
    SUPPORTED_AGGREGATION_METHODS,
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_path(path: str) -> str:
    return str(Path(path).resolve())


def _read_selected_sample_records(input_path: Path) -> List[SelectedSampleRecord]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[SelectedSampleRecord] = []
        for row in reader:
            rows.append(
                SelectedSampleRecord(
                    image_path=str(row["image_path"]),
                    label_path=str(row["label_path"]),
                    stratum_key=str(row["stratum_key"]),
                    selection_rank_within_stratum=int(row["selection_rank_within_stratum"]),
                    true_distance_m=float(row["true_distance_m"]),
                    weather=str(row["weather"]),
                    time_of_day=str(row["time_of_day"]),
                )
            )
        return rows


def _record_key(image_path: str, context_window: str, aggregation_method: str) -> Tuple[str, str, str]:
    return (_normalize_path(image_path), str(context_window), str(aggregation_method))


def _build_sample_lookup(
    dataset_root: Path,
    strict: bool,
    selected_image_paths: Iterable[str],
) -> Dict[str, Any]:
    target_paths = {_normalize_path(path) for path in selected_image_paths}
    loader = DroneDatasetLoader(dataset_root=dataset_root, strict=strict)
    lookup: Dict[str, Any] = {}
    for sample in loader.iter_samples():
        image_key = _normalize_path(str(sample.image_path))
        if image_key in target_paths:
            lookup[image_key] = sample
            if len(lookup) == len(target_paths):
                break
    missing = sorted(target_paths.difference(lookup.keys()))
    if missing:
        raise FileNotFoundError(
            "Unable to resolve {} selected images from dataset_root. First missing image: {}".format(
                len(missing),
                missing[0],
            )
        )
    return lookup


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


def run_extension(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    study_config = resolved_config.get("study", {})
    source_config = resolved_config.get("existing_feature_sources", {})
    selected_samples_csv = Path(source_config["selected_samples_csv"])
    source_record_csvs = [Path(path) for path in source_config.get("representation_records_csvs", [])]

    selected_samples = _read_selected_sample_records(selected_samples_csv)
    selected_image_paths = [row.image_path for row in selected_samples]
    context_specs = resolve_context_window_specs(study_config.get("context_windows"))
    context_order = [spec.name for spec in context_specs]
    aggregation_methods = list(study_config.get("aggregation_methods", SUPPORTED_AGGREGATION_METHODS))
    for aggregation_method in aggregation_methods:
        if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                "Unsupported aggregation_method '{}'. Expected one of {}.".format(
                    aggregation_method,
                    SUPPORTED_AGGREGATION_METHODS,
                )
            )

    existing_records_by_key, retained_counts = _load_existing_records(
        source_paths=source_record_csvs,
        selected_image_paths=selected_image_paths,
        desired_contexts=context_order,
        desired_aggregations=aggregation_methods,
    )

    total_expected_rows = len(selected_samples) * len(context_order) * len(aggregation_methods)
    missing_keys = []
    for selected_sample in selected_samples:
        image_path = selected_sample.image_path
        for context_name in context_order:
            for aggregation_method in aggregation_methods:
                key = _record_key(image_path, context_name, aggregation_method)
                if key not in existing_records_by_key:
                    missing_keys.append(key)

    if verbose:
        print(
            "[study-03] Extending feature cache\n"
            "  config: {}\n"
            "  output_root: {}\n"
            "  selected_samples_csv: {}\n"
            "  source_record_csvs: {}\n"
            "  num_selected_samples: {}\n"
            "  desired_contexts: {}\n"
            "  desired_aggregations: {}\n"
            "  existing_rows_reused: {}\n"
            "  missing_rows_to_compute: {}\n"
            "  total_expected_rows: {}".format(
                config_path or DEFAULT_CONFIG_PATH,
                output_root,
                selected_samples_csv,
                ", ".join(str(path) for path in source_record_csvs) or "(none)",
                len(selected_samples),
                ", ".join(context_order),
                ", ".join(aggregation_methods),
                len(existing_records_by_key),
                len(missing_keys),
                total_expected_rows,
            )
        )

    new_records: List[DepthRepresentationRecord] = []
    if missing_keys:
        sample_lookup = _build_sample_lookup(
            dataset_root=Path(resolved_config["dataset_root"]),
            strict=bool(resolved_config.get("strict", True)),
            selected_image_paths=selected_image_paths,
        )
        depth_config = resolved_config.get("depth_model", {})
        if verbose:
            print(
                "[study-03] Loading depth model\n"
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

        for sample_index, selected_sample in enumerate(selected_samples, start=1):
            image_key = _normalize_path(selected_sample.image_path)
            sample = sample_lookup[image_key]
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

            if verbose:
                print(
                    "[study-03] Sample {}/{} | computing missing contexts: {}\n"
                    "  image: {}".format(
                        sample_index,
                        len(selected_samples),
                        ", ".join(spec.name for spec, _ in missing_context_specs),
                        selected_sample.image_path,
                    )
                )

            image = DroneDatasetLoader.load_image(sample)
            for context_spec, missing_aggs in missing_context_specs:
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
                    key = _record_key(
                        record.image_path,
                        record.context_window,
                        record.aggregation_method,
                    )
                    existing_records_by_key[key] = record
                    new_records.append(record)
                if verbose:
                    print(
                        "[study-03]   context {} complete | added {} record(s)".format(
                            context_spec.name,
                            len(missing_aggs),
                        )
                    )

    sample_order = {_normalize_path(row.image_path): index for index, row in enumerate(selected_samples)}
    context_rank = {name: index for index, name in enumerate(context_order)}
    aggregation_rank = {name: index for index, name in enumerate(aggregation_methods)}
    all_records = sorted(
        existing_records_by_key.values(),
        key=lambda record: (
            sample_order[_normalize_path(record.image_path)],
            context_rank[record.context_window],
            aggregation_rank[record.aggregation_method],
        ),
    )

    features_csv_path = write_depth_representation_records_csv(
        all_records,
        features_dir / "representation_records.csv",
    )
    selected_samples_output_path = write_selected_sample_records_csv(
        selected_samples,
        features_dir / "selected_samples.csv",
    )

    summary = summarize_depth_representation_records(all_records, selected_samples)
    summary.update(
        {
            "study_name": resolved_config.get("study_name", output_root.name),
            "dataset_root": str(resolved_config["dataset_root"]),
            "sample_count_per_stratum_requested": int(study_config.get("sample_count_per_stratum", 10)),
            "sample_seed": int(study_config.get("sample_seed", 0)),
            "representation_records_csv": str(features_csv_path),
            "selected_samples_csv": str(selected_samples_output_path),
            "source_selected_samples_csv": str(selected_samples_csv),
            "source_representation_records_csvs": [str(path) for path in source_record_csvs],
            "retained_record_counts_by_source": retained_counts,
            "num_reused_records": len(all_records) - len(new_records),
            "num_new_records": len(new_records),
            "total_expected_rows": total_expected_rows,
            "total_actual_rows": len(all_records),
            "missing_rows_after_extension": max(0, total_expected_rows - len(all_records)),
        }
    )
    summary_path = features_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(
            "[study-03] Feature cache extension complete\n"
            "  selected_samples_csv: {}\n"
            "  representation_records_csv: {}\n"
            "  summary_json: {}\n"
            "  num_reused_records: {}\n"
            "  num_new_records: {}\n"
            "  total_rows: {}".format(
                selected_samples_output_path,
                features_csv_path,
                summary_path,
                len(all_records) - len(new_records),
                len(new_records),
                len(all_records),
            )
        )

    return {
        "selected_samples_csv": str(selected_samples_output_path),
        "representation_records_csv": str(features_csv_path),
        "summary_json": str(summary_path),
        "num_reused_records": len(all_records) - len(new_records),
        "num_new_records": len(new_records),
        "total_rows": len(all_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output and print only the final JSON result.",
    )
    args = parser.parse_args()
    result = run_extension(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
