"""Extract Attempt1-style features for source images missing from Attempt 8.

This script completes the Attempt1 feature table for the full Attempt 8 source
universe without changing the original Attempt1 artifact. It uses the same
Depth Anything V2 + bbox feature extraction path used by Attempt1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.features import (  # noqa: E402
    RegressionRecord,
    _build_record,
    read_regression_records_csv,
    write_regression_records_csv,
)
from pipeline.data.loader import (  # noqa: E402
    DroneDatasetLoader,
    DroneDatasetSample,
    _read_image_size,
    extract_metadata_from_filename,
    parse_yolo_label_file,
)
from pipeline.depth.depth_model import (  # noqa: E402
    estimate_relative_depth,
    load_depth_anything_v2,
)
from pipeline.depth.drone_depth import extract_drone_relative_depth  # noqa: E402


DEFAULT_ATTEMPT8_CONFIG = REPO_ROOT / "attempts/attempt8-comprehensive run/studies/study_all/config.yaml"
DEFAULT_ATTEMPT1_CONFIG = REPO_ROOT / "attempts/attempt1/config.yaml"
DEFAULT_BASE_ATTEMPT1_FEATURES = REPO_ROOT / "attempts/attempt1/artifacts/features/features.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "attempts/attempt8-comprehensive run/studies/study_all/artifacts/features"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_image_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).replace("\\", "/").strip().lower()


def _read_csv_image_ids(path: Path, column: str = "image_path") -> set[str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain required column '{column}'.")
        return {
            _normalize_image_id(row.get(column))
            for row in reader
            if _normalize_image_id(row.get(column))
        }


def _read_csv_image_id_map(path: Path, column: str = "image_path") -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain required column '{column}'.")
        mapping: dict[str, str] = {}
        for row in reader:
            raw_path = str(row.get(column, "")).strip()
            image_id = _normalize_image_id(raw_path)
            if image_id and image_id not in mapping:
                mapping[image_id] = raw_path
        return mapping


def _attempt4_split_source_ids(expanded_csv: Path, split_json: Path) -> set[str]:
    with expanded_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "source_group_id" not in reader.fieldnames:
            raise ValueError(
                f"{expanded_csv} does not contain required column 'source_group_id'."
            )
        source_group_ids = [row["source_group_id"] for row in reader]

    with split_json.open("r", encoding="utf-8") as fh:
        split = json.load(fh)

    indices = list(split.get("dev_indices", [])) + list(split.get("test_indices", []))
    if not indices:
        raise ValueError(f"{split_json} does not contain dev_indices/test_indices.")

    max_index = len(source_group_ids) - 1
    bad_indices = [idx for idx in indices if idx < 0 or idx > max_index]
    if bad_indices:
        preview = bad_indices[:10]
        raise ValueError(f"{split_json} references indices outside {expanded_csv}: {preview}")

    return {
        _normalize_image_id(source_group_ids[idx])
        for idx in indices
        if _normalize_image_id(source_group_ids[idx])
    }


def _target_attempt8_source_ids(
    attempt8_config: dict[str, Any],
) -> tuple[set[str], dict[str, int], dict[str, str]]:
    sources = attempt8_config["sources"]

    split_ids = _attempt4_split_source_ids(
        _repo_path(sources["attempt4_expanded_feature_table_csv"]),
        _repo_path(sources["attempt4_test_split_json"]),
    )
    aggregated_ids = _read_csv_image_ids(_repo_path(sources["aggregated_feature_table_csv"]))
    study06_id_map = _read_csv_image_id_map(_repo_path(sources["study06_fused_features_csv"]))
    study06_ids = set(study06_id_map)

    target_ids = split_ids & aggregated_ids & study06_ids
    target_path_by_id = {image_id: study06_id_map[image_id] for image_id in target_ids}
    counts = {
        "attempt4_split_source_ids": len(split_ids),
        "aggregated_source_ids": len(aggregated_ids),
        "study06_source_ids": len(study06_ids),
        "target_source_ids": len(target_ids),
    }
    return target_ids, counts, target_path_by_id


def _record_sort_key(record: RegressionRecord) -> tuple[float, str, str, str]:
    return (
        float(record.true_distance_m),
        str(record.weather),
        str(record.time_of_day),
        _normalize_image_id(record.image_path),
    )


def _load_existing_records(path: Path) -> tuple[list[RegressionRecord], set[str]]:
    records = read_regression_records_csv(path)
    ids = {_normalize_image_id(record.image_path) for record in records}
    if len(ids) != len(records):
        raise ValueError(f"{path} contains duplicate normalized image_path values.")
    return records, ids


def _collect_missing_samples(
    *,
    attempt1_config: dict[str, Any],
    missing_paths: list[str],
) -> tuple[list[DroneDatasetSample], list[str]]:
    strict = bool(attempt1_config.get("strict", True))
    samples: list[DroneDatasetSample] = []
    unresolved: list[str] = []

    for missing_path in missing_paths:
        image_path = Path(missing_path)
        label_path = image_path.with_suffix(".txt")

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing YOLO label for image: {image_path}")

            image_width, image_height = _read_image_size(image_path)
            true_distance_m, weather, time_of_day = extract_metadata_from_filename(image_path)
            annotation = parse_yolo_label_file(
                label_path=label_path,
                image_width=image_width,
                image_height=image_height,
            )
            samples.append(
                DroneDatasetSample(
                    image_path=image_path,
                    label_path=label_path,
                    true_distance_m=true_distance_m,
                    weather=weather,
                    time_of_day=time_of_day,
                    image_width=image_width,
                    image_height=image_height,
                    annotation=annotation,
                )
            )
        except (OSError, ValueError) as exc:
            if strict:
                raise
            unresolved.append(f"{missing_path}: {exc}")

    return samples, unresolved


def _extract_missing_records(
    *,
    samples: list[DroneDatasetSample],
    attempt1_config: dict[str, Any],
    local_files_only: bool,
    progress_every: int,
) -> list[RegressionRecord]:
    model_cfg = dict(attempt1_config.get("depth_model", {}))
    device = model_cfg.get("device")

    print(
        "Loading depth model "
        f"{model_cfg.get('model_name', 'depth-anything/Depth-Anything-V2-Small-hf')} "
        f"on {device or 'auto'}..."
    )
    depth_model = load_depth_anything_v2(
        model_name=model_cfg.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf"),
        device=device,
        local_files_only=local_files_only,
    )

    dataset_root = _repo_path(attempt1_config["dataset_root"])
    strict = bool(attempt1_config.get("strict", True))
    loader = DroneDatasetLoader(dataset_root, strict=strict)

    records: list[RegressionRecord] = []
    started = time.time()
    total = len(samples)

    for idx, sample in enumerate(samples, start=1):
        image = loader.load_image(sample)
        depth_map = estimate_relative_depth(image, depth_model)
        relative_depth = extract_drone_relative_depth(depth_map, sample.annotation.bbox)
        records.append(_build_record(sample, relative_depth))

        if idx == total or idx % progress_every == 0:
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed else 0.0
            remaining = (total - idx) / rate if rate else 0.0
            print(
                f"Extracted {idx}/{total} missing rows "
                f"({elapsed / 60:.1f} min elapsed, {remaining / 60:.1f} min ETA)."
            )

    return records


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Attempt1-style depth/bbox features for the source images "
            "missing from Attempt8's common data universe."
        )
    )
    parser.add_argument("--attempt8-config", type=Path, default=DEFAULT_ATTEMPT8_CONFIG)
    parser.add_argument("--attempt1-config", type=Path, default=DEFAULT_ATTEMPT1_CONFIG)
    parser.add_argument(
        "--base-attempt1-features-csv",
        type=Path,
        default=DEFAULT_BASE_ATTEMPT1_FEATURES,
        help=(
            "Existing Attempt1-style feature CSV to complete. Defaults to the original "
            "Attempt1 artifact, even if Attempt8 config already points at a completed CSV."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "attempt1_features_completed.csv",
        help="Combined existing Attempt1 rows plus newly extracted missing rows.",
    )
    parser.add_argument(
        "--missing-output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "attempt1_missing_features_only.csv",
        help="Only the newly extracted missing rows.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "attempt1_missing_feature_extraction_summary.json",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=None,
        help="Extract at most N missing rows. Useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what is missing; do not load the depth model or write CSV files.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default the script uses local cached model files only.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print extraction progress every N images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attempt8_config_path = _repo_path(args.attempt8_config)
    attempt1_config_path = _repo_path(args.attempt1_config)
    base_attempt1_features_csv = _repo_path(args.base_attempt1_features_csv)
    output_csv = _repo_path(args.output_csv)
    missing_output_csv = _repo_path(args.missing_output_csv)
    summary_json = _repo_path(args.summary_json)

    attempt8_config = _load_yaml(attempt8_config_path)
    attempt1_config = _load_yaml(attempt1_config_path)

    configured_attempt1_features_csv = _repo_path(attempt8_config["sources"]["attempt1_features_csv"])
    target_ids, coverage_counts, target_path_by_id = _target_attempt8_source_ids(attempt8_config)
    existing_records, existing_ids = _load_existing_records(base_attempt1_features_csv)

    missing_ids = sorted(target_ids - existing_ids)
    selected_missing_ids = missing_ids
    if args.max_missing is not None:
        selected_missing_ids = selected_missing_ids[: args.max_missing]
    selected_missing_paths = [target_path_by_id[image_id] for image_id in selected_missing_ids]

    samples, unresolved_ids = _collect_missing_samples(
        attempt1_config=attempt1_config,
        missing_paths=selected_missing_paths,
    )

    summary: dict[str, Any] = {
        "attempt8_config": str(attempt8_config_path),
        "attempt1_config": str(attempt1_config_path),
        "base_attempt1_features_csv": str(base_attempt1_features_csv),
        "configured_attempt8_attempt1_features_csv": str(configured_attempt1_features_csv),
        **coverage_counts,
        "existing_attempt1_rows": len(existing_records),
        "missing_attempt1_rows_total": len(missing_ids),
        "missing_attempt1_rows_selected": len(selected_missing_ids),
        "missing_samples_resolved": len(samples),
        "missing_samples_unresolved": len(unresolved_ids),
        "first_missing_image_paths": [target_path_by_id[image_id] for image_id in missing_ids[:25]],
        "first_unresolved_image_paths": unresolved_ids[:25],
        "dry_run": bool(args.dry_run),
        "local_files_only": not bool(args.allow_download),
        "output_csv": str(output_csv),
        "missing_output_csv": str(missing_output_csv),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if unresolved_ids:
        raise RuntimeError(
            "Some missing image ids were not found by DroneDatasetLoader. "
            f"First unresolved ids: {unresolved_ids[:10]}"
        )

    if args.dry_run:
        _write_summary(summary_json, summary)
        print(f"Dry run summary written to {summary_json}")
        return

    if not selected_missing_ids:
        print("No missing rows to extract. Writing a completed CSV that matches the existing Attempt1 table.")
        missing_records: list[RegressionRecord] = []
    else:
        missing_records = _extract_missing_records(
            samples=samples,
            attempt1_config=attempt1_config,
            local_files_only=not bool(args.allow_download),
            progress_every=max(1, int(args.progress_every)),
        )

    missing_by_id = {_normalize_image_id(record.image_path): record for record in missing_records}
    if len(missing_by_id) != len(missing_records):
        raise ValueError("Newly extracted records contain duplicate normalized image_path values.")

    combined_by_id = {_normalize_image_id(record.image_path): record for record in existing_records}
    overlap = sorted(set(combined_by_id) & set(missing_by_id))
    if overlap:
        raise ValueError(f"Newly extracted rows overlap existing Attempt1 rows: {overlap[:10]}")
    combined_by_id.update(missing_by_id)

    completed_records = sorted(combined_by_id.values(), key=_record_sort_key)
    missing_records = sorted(missing_records, key=_record_sort_key)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    missing_output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_regression_records_csv(missing_records, missing_output_csv)
    write_regression_records_csv(completed_records, output_csv)

    summary.update(
        {
            "new_rows_written": len(missing_records),
            "completed_rows_written": len(completed_records),
            "summary_json": str(summary_json),
        }
    )
    _write_summary(summary_json, summary)

    print(f"Wrote missing-only features to {missing_output_csv}")
    print(f"Wrote completed Attempt1 feature table to {output_csv}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
