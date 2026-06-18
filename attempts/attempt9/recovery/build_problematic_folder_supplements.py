"""
Build separate recovery feature tables for Attempt9 problematic folders.

This script intentionally does not merge into the clean Attempt9 datasets. It
recomputes all rows from the images and labels currently present on disk, so we
can inspect the recovered folders before deciding how to merge them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt9.study02.run_extract_fused_features import (  # noqa: E402
    _compute_depth_features as _compute_fused_depth_features,
    _depth_feature_names as _fused_depth_feature_names,
)
from attempts.attempt9.study03.run_extract_noisy_aggregated_features import (  # noqa: E402
    _aggregate_numeric_values,
    _compute_depth_features as _compute_noisy_depth_features,
    _depth_feature_names as _noisy_depth_feature_names,
    _jitter_bbox,
    _resolve_jitter_specs,
    _round_float,
)
from pipeline.calibration.features import (  # noqa: E402
    REGRESSION_RECORD_FIELDNAMES,
    _build_record,
    read_regression_records_csv,
    write_regression_records_csv,
)
from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES  # noqa: E402
from pipeline.data.loader import (  # noqa: E402
    DroneDatasetLoader,
    DroneDatasetSample,
    IMAGE_EXTENSION,
    _read_image_size,
    extract_metadata_from_filename,
    load_image_rgb,
    parse_yolo_label_file,
)
from pipeline.depth.depth_model import estimate_depth_map, load_depth_pro  # noqa: E402
from pipeline.depth.drone_depth import extract_drone_relative_depth  # noqa: E402
from pipeline.depth.representation import resolve_context_window_specs  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_SOURCE_FUSED_CSV = (
    REPO_ROOT
    / "attempts/attempt2/studies/study_06_all_images_midpoint_random_forest/artifacts/fused_features/depth_only_feature_table.csv"
)
DEFAULT_SOURCE_AGGREGATED_CSV = (
    REPO_ROOT / "attempts/attempt3/studies/study03/artifacts/features/noisy_aggregated_feature_table.csv"
)


def _resolve_repo_path(value: Optional[str | Path], fallback: Path) -> Path:
    if value is None:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _write_csv_rows(rows: Iterable[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv_row(row: dict[str, Any], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _normalize_path(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _sample_key(sample: DroneDatasetSample) -> Tuple[float, str, str, str]:
    return (
        float(sample.true_distance_m),
        str(sample.weather),
        str(sample.time_of_day),
        str(sample.image_path),
    )


def _stratum_key(sample: DroneDatasetSample) -> str:
    return "{}|{}|{}".format(
        _round_float(sample.true_distance_m),
        str(sample.weather),
        str(sample.time_of_day),
    )


def _reconstruct_sample_from_image(image_path: Path) -> DroneDatasetSample:
    label_path = image_path.with_suffix(".txt")
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


def _load_target_samples(
    dataset_root: Path,
    problematic_path_fragments: Sequence[str],
    strict: bool,
) -> tuple[list[DroneDatasetSample], list[dict[str, Any]]]:
    samples: list[DroneDatasetSample] = []
    skipped: list[dict[str, Any]] = []
    for fragment in problematic_path_fragments:
        folder = dataset_root / Path(fragment)
        if not folder.exists():
            skipped.append({"path": str(folder), "reason": "problematic_folder_not_found"})
            if strict:
                raise FileNotFoundError("Problematic folder not found: {}".format(folder))
            continue
        for image_path in sorted(folder.glob("*{}".format(IMAGE_EXTENSION))):
            try:
                samples.append(_reconstruct_sample_from_image(image_path))
            except (FileNotFoundError, ValueError, OSError) as exc:
                skipped.append({"path": str(image_path), "reason": str(exc)})
                if strict:
                    raise
    return sorted(samples, key=_sample_key), skipped


def _build_selection_ranks(samples: Sequence[DroneDatasetSample]) -> dict[str, int]:
    grouped: dict[tuple[float, str, str], list[DroneDatasetSample]] = defaultdict(list)
    for sample in samples:
        grouped[(float(sample.true_distance_m), str(sample.weather), str(sample.time_of_day))].append(sample)
    ranks: dict[str, int] = {}
    for stratum in sorted(grouped):
        for rank, sample in enumerate(sorted(grouped[stratum], key=lambda item: str(item.image_path)), start=1):
            ranks[_normalize_path(sample.image_path)] = rank
    return ranks


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


def _study02_base_row(sample: DroneDatasetSample, selection_ranks: dict[str, int]) -> dict[str, Any]:
    return {
        "image_path": str(sample.image_path),
        "label_path": str(sample.label_path),
        "stratum_key": _stratum_key(sample),
        "selection_rank_within_stratum": int(selection_ranks[_normalize_path(sample.image_path)]),
        "true_distance_m": _round_float(sample.true_distance_m),
        "weather": str(sample.weather),
        "time_of_day": str(sample.time_of_day),
    }


def _study03_base_row(
    sample: DroneDatasetSample,
    jitter_specs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    jittered_bboxes = [
        _jitter_bbox(
            bbox=sample.annotation.bbox,
            image_width=int(sample.image_width),
            image_height=int(sample.image_height),
            jitter_spec=jitter_spec,
        )
        for jitter_spec in jitter_specs
    ]
    row: dict[str, Any] = {
        "image_path": str(sample.image_path),
        "label_path": str(sample.label_path),
        "stratum_key": _stratum_key(sample),
        "true_distance_m": _round_float(sample.true_distance_m),
        "weather": str(sample.weather),
        "time_of_day": str(sample.time_of_day),
        "num_jitter_boxes": int(len(jittered_bboxes)),
    }
    for feature_name in GEOMETRY_FEATURE_NAMES:
        values = [
            _build_geometry_feature_dict(
                bbox=bbox,
                image_width=int(sample.image_width),
                image_height=int(sample.image_height),
            )[feature_name]
            for bbox in jittered_bboxes
        ]
        feature_median, feature_std = _aggregate_numeric_values(values)
        row["{}__jitter_median".format(feature_name)] = _round_float(feature_median)
        row["{}__jitter_std".format(feature_name)] = _round_float(feature_std)
    return row


def _partial_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {_normalize_path(row["image_path"]) for row in _read_csv_rows(path)}


def _write_manifest(samples: Sequence[DroneDatasetSample], output_path: Path) -> None:
    rows = [
        {
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
            "stratum_key": _stratum_key(sample),
            "true_distance_m": _round_float(sample.true_distance_m),
            "weather": str(sample.weather),
            "time_of_day": str(sample.time_of_day),
            "image_width": int(sample.image_width),
            "image_height": int(sample.image_height),
            "bbox": "{} {} {} {}".format(*sample.annotation.bbox),
        }
        for sample in samples
    ]
    _write_csv_rows(
        rows,
        output_path,
        [
            "image_path",
            "label_path",
            "stratum_key",
            "true_distance_m",
            "weather",
            "time_of_day",
            "image_width",
            "image_height",
            "bbox",
        ],
    )


def _build_study01_supplement(
    samples: Sequence[DroneDatasetSample],
    output_dir: Path,
    output_filename: str,
    depth_model: Any,
    resume: bool,
    progress_every: int,
    verbose: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / output_filename
    partial_csv = output_dir / output_filename.replace(".csv", "_partial.csv")
    existing_records = read_regression_records_csv(partial_csv) if resume and partial_csv.exists() else []
    existing_keys = {_normalize_path(record.image_path) for record in existing_records}
    records = list(existing_records)
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        if _normalize_path(sample.image_path) in existing_keys:
            continue
        image = load_image_rgb(sample.image_path)
        depth_map = estimate_depth_map(image, depth_model)
        metric_depth = extract_drone_relative_depth(depth_map, sample.annotation.bbox)
        record = _build_record(sample, metric_depth)
        records.append(record)
        existing_keys.add(_normalize_path(sample.image_path))
        with partial_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGRESSION_RECORD_FIELDNAMES)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(record.to_dict())
        if verbose and (index == 1 or index == len(samples) or index % progress_every == 0):
            elapsed = time.time() - started
            print(
                "[attempt9/recovery:study01] Row {}/{} | written={} | elapsed={:.1f} min".format(
                    index,
                    len(samples),
                    len(records),
                    elapsed / 60.0,
                )
            )
    write_regression_records_csv(records, output_csv)
    return {"dataset": "study01", "output_csv": str(output_csv), "partial_csv": str(partial_csv), "rows": len(records)}


def _build_study02_supplement(
    samples: Sequence[DroneDatasetSample],
    selection_ranks: dict[str, int],
    source_csv: Path,
    config: dict[str, Any],
    output_dir: Path,
    depth_model: Any,
    resume: bool,
    progress_every: int,
    verbose: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _read_csv_fieldnames(source_csv)
    depth_columns = _fused_depth_feature_names(fieldnames)
    context_specs = list(resolve_context_window_specs(config.get("context_windows")))
    aggregation_methods = list(config.get("aggregation_methods", []))
    score_fields = list(config.get("score_fields", ["object_depth"]))
    output_csv = output_dir / str(config.get("output_filename", "depth_only_feature_table_problematic_folders.csv"))
    partial_csv = output_dir / output_csv.name.replace(".csv", "_partial.csv")
    rows = _read_csv_rows(partial_csv) if resume and partial_csv.exists() else []
    existing_keys = _partial_keys(partial_csv) if resume else set()
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        if _normalize_path(sample.image_path) in existing_keys:
            continue
        row = _study02_base_row(sample, selection_ranks)
        computed = _compute_fused_depth_features(
            row=row,
            depth_model=depth_model,
            context_specs=context_specs,
            aggregation_methods=aggregation_methods,
            score_fields=score_fields,
            inner_bbox_scale=float(config.get("inner_bbox_scale", 0.5)),
            surrounding_bbox_scale=float(config.get("surrounding_bbox_scale", 1.8)),
        )
        for name in depth_columns:
            row[name] = computed[name]
        rows.append(row)
        existing_keys.add(_normalize_path(sample.image_path))
        _append_csv_row(row, partial_csv, fieldnames)
        if verbose and (index == 1 or index == len(samples) or index % progress_every == 0):
            elapsed = time.time() - started
            print(
                "[attempt9/recovery:study02] Row {}/{} | written={} | elapsed={:.1f} min".format(
                    index,
                    len(samples),
                    len(rows),
                    elapsed / 60.0,
                )
            )
    _write_csv_rows(rows, output_csv, fieldnames)
    return {
        "dataset": "study02",
        "output_csv": str(output_csv),
        "partial_csv": str(partial_csv),
        "rows": len(rows),
        "depth_columns_recomputed": depth_columns,
    }


def _build_study03_supplement(
    samples: Sequence[DroneDatasetSample],
    source_csv: Path,
    config: dict[str, Any],
    output_dir: Path,
    depth_model: Any,
    resume: bool,
    progress_every: int,
    verbose: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _read_csv_fieldnames(source_csv)
    depth_columns = _noisy_depth_feature_names(fieldnames)
    context_specs = list(resolve_context_window_specs(config.get("context_windows")))
    aggregation_methods = list(config.get("aggregation_methods", []))
    jitter_specs = _resolve_jitter_specs(config)
    output_csv = output_dir / str(config.get("output_filename", "noisy_aggregated_feature_table_problematic_folders.csv"))
    partial_csv = output_dir / output_csv.name.replace(".csv", "_partial.csv")
    rows = _read_csv_rows(partial_csv) if resume and partial_csv.exists() else []
    existing_keys = _partial_keys(partial_csv) if resume else set()
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        if _normalize_path(sample.image_path) in existing_keys:
            continue
        row = _study03_base_row(sample, jitter_specs)
        computed = _compute_noisy_depth_features(
            row=row,
            depth_model=depth_model,
            context_specs=context_specs,
            aggregation_methods=aggregation_methods,
            jitter_specs=jitter_specs,
            inner_bbox_scale=float(config.get("inner_bbox_scale", 0.5)),
            surrounding_bbox_scale=float(config.get("surrounding_bbox_scale", 1.8)),
        )
        for name in depth_columns:
            row[name] = computed[name]
        rows.append(row)
        existing_keys.add(_normalize_path(sample.image_path))
        _append_csv_row(row, partial_csv, fieldnames)
        if verbose and (index == 1 or index == len(samples) or index % progress_every == 0):
            elapsed = time.time() - started
            print(
                "[attempt9/recovery:study03] Row {}/{} | written={} | elapsed={:.1f} min".format(
                    index,
                    len(samples),
                    len(rows),
                    elapsed / 60.0,
                )
            )
    _write_csv_rows(rows, output_csv, fieldnames)
    return {
        "dataset": "study03",
        "output_csv": str(output_csv),
        "partial_csv": str(partial_csv),
        "rows": len(rows),
        "depth_columns_recomputed": depth_columns,
    }


def run_recovery(
    config_path: Optional[Path] = None,
    datasets: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    config = _read_yaml(config_path or DEFAULT_CONFIG_PATH)
    dataset_root = _resolve_repo_path(config.get("dataset_root"), Path(r"C:\Users\depthlev\Desktop\droneImages\dataset"))
    output_root = _resolve_repo_path(config.get("output_root"), REPO_ROOT / "attempts/attempt9/recovery/artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    strict = bool(config.get("strict", True))
    run_config = dict(config.get("run", {}))
    selected_datasets = list(datasets or run_config.get("datasets", ["study01", "study02", "study03"]))
    resume = bool(run_config.get("resume_from_partial", True))
    progress_every = max(1, int(run_config.get("progress_log_every_rows", 25)))
    problematic_path_fragments = list(config.get("problematic_path_fragments", []))

    samples, skipped_samples = _load_target_samples(
        dataset_root=dataset_root,
        problematic_path_fragments=problematic_path_fragments,
        strict=strict,
    )
    max_samples = config.get("max_samples")
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    selection_ranks = _build_selection_ranks(samples)
    _write_manifest(samples, audit_dir / "problematic_folder_manifest.csv")
    _write_csv_rows(skipped_samples, audit_dir / "skipped_problematic_samples.csv", ["path", "reason"])

    if verbose:
        print(
            "[attempt9/recovery] Problematic folder recovery setup\n"
            "  dataset_root: {}\n"
            "  output_root: {}\n"
            "  problematic_folders: {}\n"
            "  target_samples: {}\n"
            "  datasets: {}\n"
            "  resume_from_partial: {}".format(
                dataset_root,
                output_root,
                problematic_path_fragments,
                len(samples),
                selected_datasets,
                resume,
            )
        )

    depth_config = dict(config.get("depth_model", {}))
    depth_model = load_depth_pro(
        checkpoint_path=_resolve_repo_path(depth_config.get("checkpoint_path"), REPO_ROOT / "checkpoints/depth_pro.pt"),
        device=depth_config.get("device"),
    )

    results = []
    if "study01" in selected_datasets:
        results.append(
            _build_study01_supplement(
                samples=samples,
                output_dir=output_root / "study01",
                output_filename=str(dict(config.get("study01", {})).get("output_filename", "features_problematic_folders.csv")),
                depth_model=depth_model,
                resume=resume,
                progress_every=progress_every,
                verbose=verbose,
            )
        )
    if "study02" in selected_datasets:
        results.append(
            _build_study02_supplement(
                samples=samples,
                selection_ranks=selection_ranks,
                source_csv=_resolve_repo_path(
                    dict(config.get("sources", {})).get("source_fused_features_csv"),
                    DEFAULT_SOURCE_FUSED_CSV,
                ),
                config=dict(config.get("study02", {})),
                output_dir=output_root / "study02",
                depth_model=depth_model,
                resume=resume,
                progress_every=progress_every,
                verbose=verbose,
            )
        )
    if "study03" in selected_datasets:
        results.append(
            _build_study03_supplement(
                samples=samples,
                source_csv=_resolve_repo_path(
                    dict(config.get("sources", {})).get("source_aggregated_features_csv"),
                    DEFAULT_SOURCE_AGGREGATED_CSV,
                ),
                config=dict(config.get("study03", {})),
                output_dir=output_root / "study03",
                depth_model=depth_model,
                resume=resume,
                progress_every=progress_every,
                verbose=verbose,
            )
        )

    summary = {
        "study_name": config.get("study_name", "attempt9_problematic_folder_recovery"),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "problematic_path_fragments": problematic_path_fragments,
        "target_samples": len(samples),
        "skipped_problematic_samples": len(skipped_samples),
        "depth_model_kind": "depth_pro",
        "depth_model_semantics": "metric_model_depth_meters",
        "datasets": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print("[attempt9/recovery] Complete\n  summary_json: {}".format(summary_path))
        print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--datasets", nargs="+", choices=["study01", "study02", "study03"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_recovery(config_path=args.config, datasets=args.datasets, verbose=not args.quiet)
    if args.quiet:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
