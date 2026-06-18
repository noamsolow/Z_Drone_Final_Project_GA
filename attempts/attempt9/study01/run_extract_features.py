"""
Attempt 9 / Study 01 feature extraction.

This rebuilds the existing Attempt1-compatible regression dataset row-for-row,
but replaces the old Depth Anything depth value with a new Depth Pro depth
measurement. All non-depth fields are reused from the base CSV so the output
remains schema-compatible with the current downstream studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.features import (  # noqa: E402
    REGRESSION_RECORD_FIELDNAMES,
    RegressionRecord,
    _build_record,
    read_regression_records_csv,
    summarize_regression_records,
    write_regression_records_csv,
)
from pipeline.data.loader import (  # noqa: E402
    DroneDatasetLoader,
    DroneDatasetSample,
    _read_image_size,
    extract_metadata_from_filename,
    parse_yolo_label_file,
)
from pipeline.depth.depth_model import estimate_depth_map, load_depth_pro  # noqa: E402
from pipeline.depth.drone_depth import extract_drone_relative_depth  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_BASE_FEATURES_CSV = REPO_ROOT / "attempts/attempt1/artifacts/features/features.csv"

NON_DEPTH_FIELD_NAMES = (
    "image_path",
    "label_path",
    "true_distance_m",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "weather",
    "time_of_day",
)


def _resolve_repo_path(value: Optional[str | Path], fallback: Path) -> Path:
    if value is None:
        return fallback

    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _format_minutes(seconds: float) -> str:
    return "{:.1f} min".format(max(0.0, seconds) / 60.0)


def _normalize_image_id(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _normalize_path_for_match(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _resolve_excluded_path_fragments(study_config: Dict[str, Any]) -> list[str]:
    return [
        _normalize_path_for_match(fragment)
        for fragment in study_config.get("excluded_path_fragments", [])
    ]


def _is_excluded_path(path_value: str, excluded_path_fragments: list[str]) -> bool:
    normalized_path = _normalize_path_for_match(path_value)
    return any(fragment in normalized_path for fragment in excluded_path_fragments)


def _load_partial_records(path: Path) -> list[RegressionRecord]:
    if not path.exists():
        return []
    return read_regression_records_csv(path)


def _append_regression_record_csv(record: RegressionRecord, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGRESSION_RECORD_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record.to_dict())


def _write_skipped_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_row_index",
        "image_path",
        "label_path",
        "reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _reconstruct_sample_from_record(record: RegressionRecord) -> DroneDatasetSample:
    image_path = Path(record.image_path)
    label_path = Path(record.label_path)

    image_width, image_height = _read_image_size(image_path)
    true_distance_m, weather, time_of_day = extract_metadata_from_filename(image_path)
    annotation = parse_yolo_label_file(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
    )

    return DroneDatasetSample(
        image_path=image_path,
        label_path=label_path,
        true_distance_m=true_distance_m,
        weather=weather,
        time_of_day=time_of_day,
        image_width=image_width,
        image_height=image_height,
        annotation=annotation,
    )


def _validate_base_record_compatibility(
    base_record: RegressionRecord,
    rebuilt_record: RegressionRecord,
) -> None:
    mismatches = []
    for field_name in NON_DEPTH_FIELD_NAMES:
        if getattr(base_record, field_name) != getattr(rebuilt_record, field_name):
            mismatches.append(field_name)

    if mismatches:
        raise ValueError(
            "Base feature row no longer matches data reconstructed from disk for '{}'. "
            "Mismatched fields: {}".format(
                base_record.image_path,
                ", ".join(mismatches),
            )
        )


def _rebuild_record_with_metric_depth(
    base_record: RegressionRecord,
    sample: DroneDatasetSample,
    metric_model_depth: float,
) -> RegressionRecord:
    rebuilt_record = _build_record(sample, metric_model_depth)
    _validate_base_record_compatibility(base_record, rebuilt_record)
    return replace(base_record, relative_depth=rebuilt_record.relative_depth)


def run_feature_extraction(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    depth_model: Any = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = _resolve_repo_path(resolved_config.get("output_root"), REPO_ROOT / "attempts/attempt9/study01/artifacts")
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sources_config = dict(resolved_config.get("sources", {}))
    base_features_csv = _resolve_repo_path(
        sources_config.get("base_features_csv"),
        DEFAULT_BASE_FEATURES_CSV,
    )

    study_config = dict(resolved_config.get("study", {}))
    progress_log_every_rows = max(1, int(study_config.get("progress_log_every_rows", 50)))
    checkpoint_every_rows = max(1, int(study_config.get("checkpoint_every_rows", 1)))
    resume_from_partial = bool(study_config.get("resume_from_partial", True))
    stop_on_missing_source = bool(study_config.get("stop_on_missing_source", True))
    excluded_path_fragments = _resolve_excluded_path_fragments(study_config)
    partial_features_csv_path = features_dir / "features_partial.csv"
    skipped_rows_csv_path = features_dir / "skipped_source_rows.csv"

    if verbose:
        print(
            "[attempt9/study01] Starting Attempt1-compatible dataset rebuild with Depth Pro\n"
            "  config: {}\n"
            "  dataset_root: {}\n"
            "  source_features_csv: {}\n"
            "  output_root: {}\n"
            "  output_depth_semantics: metric model depth (Depth Pro)\n"
            "  output_depth_column_name: relative_depth (kept for compatibility)\n"
            "  partial_features_csv: {}\n"
            "  resume_from_partial: {}\n"
            "  stop_on_missing_source: {}\n"
            "  excluded_path_fragments: {}".format(
                config_path or DEFAULT_CONFIG_PATH,
                resolved_config["dataset_root"],
                base_features_csv,
                output_root,
                partial_features_csv_path,
                resume_from_partial,
                stop_on_missing_source,
                excluded_path_fragments,
            )
        )

    if depth_model is None:
        depth_config = dict(resolved_config.get("depth_model", {}))
        checkpoint_path = _resolve_repo_path(
            depth_config.get("checkpoint_path"),
            REPO_ROOT / "checkpoints/depth_pro.pt",
        )
        if verbose:
            print(
                "[attempt9/study01] Loading Depth Pro\n"
                "  device: {}\n"
                "  checkpoint_path: {}".format(
                    depth_config.get("device", "auto"),
                    checkpoint_path,
                )
            )
        depth_model = load_depth_pro(
            checkpoint_path=checkpoint_path,
            device=depth_config.get("device"),
        )

    base_records = read_regression_records_csv(base_features_csv)
    max_samples = resolved_config.get("max_samples")
    if max_samples is not None:
        base_records = base_records[: int(max_samples)]

    if verbose:
        print(
            "[attempt9/study01] Reusing {} non-depth rows from base dataset".format(
                len(base_records)
            )
        )

    loader = DroneDatasetLoader(
        dataset_root=resolved_config["dataset_root"],
        strict=bool(resolved_config.get("strict", True)),
    )

    rebuilt_records: list[RegressionRecord] = []
    skipped_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    already_written_ids: set[str] = set()
    if resume_from_partial:
        partial_records = _load_partial_records(partial_features_csv_path)
        rebuilt_records = [
            record
            for record in partial_records
            if not _is_excluded_path(record.image_path, excluded_path_fragments)
        ]
        if len(rebuilt_records) != len(partial_records):
            write_regression_records_csv(rebuilt_records, partial_features_csv_path)
        already_written_ids = {
            _normalize_image_id(record.image_path)
            for record in rebuilt_records
        }
        if verbose and rebuilt_records:
            print(
                "[attempt9/study01] Resuming from partial feature cache\n"
                "  existing_partial_rows: {}\n"
                "  partial_features_csv: {}".format(
                    len(rebuilt_records),
                    partial_features_csv_path,
                )
            )
        if verbose and len(rebuilt_records) != len(partial_records):
            print(
                "[attempt9/study01] Removed excluded rows from partial cache\n"
                "  removed_partial_rows: {}".format(len(partial_records) - len(rebuilt_records))
            )

    started = time.time()
    total = len(base_records)

    for index, base_record in enumerate(base_records, start=1):
        if _is_excluded_path(base_record.image_path, excluded_path_fragments):
            excluded_row = {
                "source_row_index": index,
                "image_path": base_record.image_path,
                "label_path": base_record.label_path,
                "reason": "excluded_problematic_folder",
            }
            skipped_rows.append(excluded_row)
            excluded_rows.append(excluded_row)
            continue

        image_id = _normalize_image_id(base_record.image_path)
        if image_id in already_written_ids:
            if verbose and (
                index == 1
                or index == total
                or index % progress_log_every_rows == 0
            ):
                print(
                    "[attempt9/study01] Row {}/{} already exists in partial cache".format(
                        index,
                        total,
                    )
                )
            continue

        try:
            sample = _reconstruct_sample_from_record(base_record)
        except (FileNotFoundError, ValueError, OSError) as exc:
            skipped_row = {
                "source_row_index": index,
                "image_path": base_record.image_path,
                "label_path": base_record.label_path,
                "reason": str(exc),
            }
            skipped_rows.append(skipped_row)
            if verbose:
                print(
                    "[attempt9/study01] Skipping source row {}/{}\n"
                    "  image: {}\n"
                    "  reason: {}".format(
                        index,
                        total,
                        base_record.image_path,
                        exc,
                    )
                )
            if stop_on_missing_source:
                _write_skipped_rows_csv(skipped_rows, skipped_rows_csv_path)
                raise
            continue

        image = loader.load_image(sample)
        depth_map = estimate_depth_map(image, depth_model)
        metric_model_depth = extract_drone_relative_depth(depth_map, sample.annotation.bbox)
        rebuilt_record = _rebuild_record_with_metric_depth(
            base_record=base_record,
            sample=sample,
            metric_model_depth=metric_model_depth,
        )
        rebuilt_records.append(rebuilt_record)
        already_written_ids.add(image_id)

        if len(rebuilt_records) % checkpoint_every_rows == 0:
            _append_regression_record_csv(rebuilt_record, partial_features_csv_path)

        if verbose and (
            index == 1
            or index == total
            or index % progress_log_every_rows == 0
        ):
            elapsed_seconds = time.time() - started
            rows_per_second = index / elapsed_seconds if elapsed_seconds > 0 else 0.0
            eta_seconds = ((total - index) / rows_per_second) if rows_per_second > 0 else 0.0
            print(
                "[attempt9/study01] Row {}/{}\n"
                "  image: {}\n"
                "  metric_model_depth: {:.6f}\n"
                "  elapsed: {}\n"
                "  eta: {}".format(
                    index,
                    total,
                    base_record.image_path,
                    float(metric_model_depth),
                    _format_minutes(elapsed_seconds),
                    _format_minutes(eta_seconds),
                )
            )

    if skipped_rows:
        _write_skipped_rows_csv(skipped_rows, skipped_rows_csv_path)

    final_filename = "features.csv" if not skipped_rows else "features_available.csv"
    features_csv_path = write_regression_records_csv(rebuilt_records, features_dir / final_filename)
    available_features_csv_path = features_dir / "features_available.csv"
    if features_csv_path != available_features_csv_path:
        write_regression_records_csv(rebuilt_records, available_features_csv_path)
    summary = summarize_regression_records(rebuilt_records)
    summary.update(
        {
            "study_name": resolved_config.get("study_name", output_root.name),
            "dataset_root": str(resolved_config["dataset_root"]),
            "source_features_csv": str(base_features_csv),
            "features_csv": str(features_csv_path),
            "available_features_csv": str(available_features_csv_path),
            "partial_features_csv": str(partial_features_csv_path),
            "skipped_source_rows_csv": str(skipped_rows_csv_path) if skipped_rows else None,
            "num_source_rows_reused_for_non_depth_fields": len(base_records),
            "num_metric_depth_rows_written": len(rebuilt_records),
            "num_source_rows_skipped": len(skipped_rows),
            "num_source_rows_excluded": len(excluded_rows),
            "excluded_path_fragments": excluded_path_fragments,
            "is_complete_attempt1_compatible_dataset": len(skipped_rows) == 0,
            "depth_model_kind": "depth_pro",
            "depth_model_semantics": "metric_model_depth_meters",
            "output_depth_column_name": "relative_depth",
            "schema_compatible_with_source_features_csv": True,
            "stop_on_missing_source": stop_on_missing_source,
            "resume_from_partial": resume_from_partial,
        }
    )
    summary_path = features_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(
            "[attempt9/study01] Feature extraction complete\n"
            "  features_csv: {}\n"
            "  summary_json: {}\n"
            "  partial_features_csv: {}\n"
            "  num_records: {}\n"
            "  skipped_source_rows: {}\n"
            "  excluded_source_rows: {}".format(
                features_csv_path,
                summary_path,
                partial_features_csv_path,
                len(rebuilt_records),
                len(skipped_rows),
                len(excluded_rows),
            )
        )

    return {
        "features_csv": str(features_csv_path),
        "partial_features_csv": str(partial_features_csv_path),
        "summary_json": str(summary_path),
        "num_records": len(rebuilt_records),
        "num_skipped_source_rows": len(skipped_rows),
        "num_excluded_source_rows": len(excluded_rows),
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
    result = run_feature_extraction(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
