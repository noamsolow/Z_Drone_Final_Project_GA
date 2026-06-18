"""
Build an Attempt3/Study03-compatible noisy aggregated feature table with Depth Pro.

The output keeps the original wide-table schema used by Attempt8. Metadata and
geometry jitter columns are reused from the source table, while every
``...__object_depth__jitter_*`` column is recomputed from Apple's Depth Pro
metric depth maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES  # noqa: E402
from pipeline.data.loader import (  # noqa: E402
    DroneDatasetSample,
    _read_image_size,
    extract_metadata_from_filename,
    load_image_rgb,
    parse_yolo_label_file,
)
from pipeline.depth.depth_model import estimate_depth_map, load_depth_pro  # noqa: E402
from pipeline.depth.representation import (  # noqa: E402
    SUPPORTED_AGGREGATION_METHODS,
    clamp_bbox,
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_SOURCE_AGGREGATED_CSV = (
    REPO_ROOT / "attempts/attempt3/studies/study03/artifacts/features/noisy_aggregated_feature_table.csv"
)
META_COLUMNS = {
    "image_path",
    "label_path",
    "stratum_key",
    "true_distance_m",
    "weather",
    "time_of_day",
    "num_jitter_boxes",
}


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


def _format_minutes(seconds: float) -> str:
    return "{:.1f} min".format(max(0.0, seconds) / 60.0)


def _round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _reconstruct_sample(row: dict[str, Any]) -> DroneDatasetSample:
    image_path = Path(str(row["image_path"]))
    label_path = Path(str(row["label_path"]))
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


def _resolve_jitter_specs(study_config: Dict[str, Any]) -> list[dict[str, Any]]:
    raw_specs = list(study_config.get("jitter_boxes", []))
    if not raw_specs:
        raise ValueError("study.jitter_boxes must not be empty.")
    resolved = []
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
    return clamp_bbox(
        (
            shifted_center_x - max(1.0, width * scale) / 2.0,
            shifted_center_y - max(1.0, height * scale) / 2.0,
            shifted_center_x + max(1.0, width * scale) / 2.0,
            shifted_center_y + max(1.0, height * scale) / 2.0,
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
        (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1),
        image_width=crop_width,
        image_height=crop_height,
    )


def _aggregate_numeric_values(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot aggregate an empty value list.")
    return float(np.median(array)), float(np.std(array))


def _depth_feature_names(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if "__object_depth__jitter_" in name]


def _load_partial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {_normalize_image_id(row["image_path"]) for row in _read_csv_rows(path)}


def _compute_depth_features(
    row: dict[str, Any],
    depth_model: Any,
    context_specs: list[Any],
    aggregation_methods: list[str],
    jitter_specs: list[dict[str, Any]],
    inner_bbox_scale: float,
    surrounding_bbox_scale: float,
) -> dict[str, float]:
    sample = _reconstruct_sample(row)
    image = load_image_rgb(sample.image_path)
    jittered_global_bboxes = [
        _jitter_bbox(
            bbox=sample.annotation.bbox,
            image_width=int(sample.image_width),
            image_height=int(sample.image_height),
            jitter_spec=jitter_spec,
        )
        for jitter_spec in jitter_specs
    ]

    computed: dict[str, float] = {}
    for context_spec in context_specs:
        crop_data = extract_context_crop(
            image=image,
            bbox=sample.annotation.bbox,
            context_spec=context_spec,
        )
        depth_map = estimate_depth_map(crop_data["image"], depth_model)
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
            if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
                raise ValueError("Unsupported aggregation method: {}".format(aggregation_method))
            values = []
            for jittered_local_bbox in local_jittered_bboxes:
                representation = compute_depth_representation(
                    depth_map=depth_map,
                    bbox=jittered_local_bbox,
                    aggregation_method=aggregation_method,
                    inner_bbox_scale=inner_bbox_scale,
                    surrounding_bbox_scale=surrounding_bbox_scale,
                )
                values.append(float(representation["object_depth"]))
            feature_median, feature_std = _aggregate_numeric_values(values)
            prefix = "{}__{}__object_depth".format(context_spec.name, aggregation_method)
            computed[prefix + "__jitter_median"] = _round_float(feature_median)
            computed[prefix + "__jitter_std"] = _round_float(feature_std)
    return computed


def run_feature_extraction(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    depth_model: Any = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or _read_yaml(config_path or DEFAULT_CONFIG_PATH))
    output_root = _resolve_repo_path(
        resolved_config.get("output_root"),
        REPO_ROOT / "attempts/attempt9/study03/artifacts",
    )
    features_dir = output_root / "features"
    audit_dir = output_root / "audit"
    features_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    source_csv = _resolve_repo_path(
        dict(resolved_config.get("sources", {})).get("source_aggregated_features_csv"),
        DEFAULT_SOURCE_AGGREGATED_CSV,
    )
    source_rows = _read_csv_rows(source_csv)
    max_samples = resolved_config.get("max_samples")
    if max_samples is not None:
        source_rows = source_rows[: int(max_samples)]

    source_fieldnames = _read_csv_fieldnames(source_csv)
    depth_columns = _depth_feature_names(source_fieldnames)
    reusable_columns = [name for name in source_fieldnames if name not in depth_columns]
    missing_geometry = [
        "{}__jitter_median".format(name)
        for name in GEOMETRY_FEATURE_NAMES
        if "{}__jitter_median".format(name) not in source_fieldnames
    ]
    if missing_geometry:
        raise ValueError("Source aggregated table is missing geometry columns: {}".format(missing_geometry))

    study_config = dict(resolved_config.get("study", {}))
    context_specs = list(resolve_context_window_specs(study_config.get("context_windows")))
    aggregation_methods = list(study_config.get("aggregation_methods", []))
    jitter_specs = _resolve_jitter_specs(study_config)
    progress_every = max(1, int(study_config.get("progress_log_every_rows", 25)))
    resume_from_partial = bool(study_config.get("resume_from_partial", True))
    stop_on_missing_source = bool(study_config.get("stop_on_missing_source", False))
    excluded_path_fragments = _resolve_excluded_path_fragments(study_config)
    inner_bbox_scale = float(study_config.get("inner_bbox_scale", 0.5))
    surrounding_bbox_scale = float(study_config.get("surrounding_bbox_scale", 1.8))

    output_csv = features_dir / "noisy_aggregated_feature_table.csv"
    partial_csv = features_dir / "noisy_aggregated_feature_table_partial.csv"
    skipped_csv = audit_dir / "skipped_source_rows.csv"
    summary_json = features_dir / "summary.json"

    if depth_model is None:
        depth_config = dict(resolved_config.get("depth_model", {}))
        depth_model = load_depth_pro(
            checkpoint_path=_resolve_repo_path(
                depth_config.get("checkpoint_path"),
                REPO_ROOT / "checkpoints/depth_pro.pt",
            ),
            device=depth_config.get("device"),
        )

    if resume_from_partial and partial_csv.exists():
        partial_rows = _read_csv_rows(partial_csv)
        completed_rows = [
            row
            for row in partial_rows
            if not _is_excluded_path(row["image_path"], excluded_path_fragments)
        ]
        if len(completed_rows) != len(partial_rows):
            _write_csv_rows(completed_rows, partial_csv, source_fieldnames)
    else:
        partial_rows = []
        completed_rows = []
    already_written = {
        _normalize_image_id(row["image_path"])
        for row in completed_rows
    }
    skipped_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(source_rows)

    if verbose:
        print(
            "[attempt9/study03] Building Study03 noisy aggregated table with Depth Pro\n"
            "  source_csv: {}\n"
            "  output_csv: {}\n"
            "  partial_csv: {}\n"
            "  source_rows: {}\n"
            "  reusable_non_depth_columns: {}\n"
            "  depth_columns_to_recompute: {}\n"
            "  context_windows: {}\n"
            "  jitter_boxes: {}\n"
            "  excluded_path_fragments: {}".format(
                source_csv,
                output_csv,
                partial_csv,
                total,
                len(reusable_columns),
                len(depth_columns),
                len(context_specs),
                len(jitter_specs),
                excluded_path_fragments,
            )
        )
        if len(completed_rows) != len(partial_rows):
            print(
                "[attempt9/study03] Removed excluded rows from partial cache\n"
                "  removed_partial_rows: {}".format(len(partial_rows) - len(completed_rows))
            )

    for index, source_row in enumerate(source_rows, start=1):
        if _is_excluded_path(source_row["image_path"], excluded_path_fragments):
            excluded_row = {
                "source_row_index": index,
                "image_path": source_row.get("image_path", ""),
                "label_path": source_row.get("label_path", ""),
                "reason": "excluded_problematic_folder",
            }
            skipped_rows.append(excluded_row)
            excluded_rows.append(excluded_row)
            continue

        image_id = _normalize_image_id(source_row["image_path"])
        if image_id in already_written:
            continue
        try:
            computed_depth = _compute_depth_features(
                row=source_row,
                depth_model=depth_model,
                context_specs=context_specs,
                aggregation_methods=aggregation_methods,
                jitter_specs=jitter_specs,
                inner_bbox_scale=inner_bbox_scale,
                surrounding_bbox_scale=surrounding_bbox_scale,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            skipped_rows.append(
                {
                    "source_row_index": index,
                    "image_path": source_row.get("image_path", ""),
                    "label_path": source_row.get("label_path", ""),
                    "reason": str(exc),
                }
            )
            if verbose:
                print(
                    "[attempt9/study03] Skipping source row {}/{}\n"
                    "  image: {}\n"
                    "  reason: {}".format(index, total, source_row.get("image_path", ""), exc)
                )
            if stop_on_missing_source:
                break
            continue

        output_row: dict[str, Any] = dict(source_row)
        for name in depth_columns:
            output_row[name] = computed_depth[name]

        completed_rows.append(output_row)
        already_written.add(image_id)
        _append_csv_row(output_row, partial_csv, source_fieldnames)

        if verbose and (index == 1 or index == total or index % progress_every == 0):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (total - index) / rate if rate > 0 else 0.0
            print(
                "[attempt9/study03] Row {}/{} | written={} | skipped={} | elapsed={} | eta={}".format(
                    index,
                    total,
                    len(completed_rows),
                    len(skipped_rows),
                    _format_minutes(elapsed),
                    _format_minutes(eta),
                )
            )

    _write_csv_rows(completed_rows, output_csv, source_fieldnames)
    _write_csv_rows(skipped_rows, skipped_csv, ["source_row_index", "image_path", "label_path", "reason"])

    summary = {
        "study_name": resolved_config.get("study_name", output_root.name),
        "source_aggregated_features_csv": str(source_csv),
        "aggregated_features_csv": str(output_csv),
        "partial_aggregated_features_csv": str(partial_csv),
        "skipped_source_rows_csv": str(skipped_csv),
        "source_rows": total,
        "rows_written": len(completed_rows),
        "rows_skipped": len(skipped_rows),
        "rows_excluded": len(excluded_rows),
        "excluded_path_fragments": excluded_path_fragments,
        "num_jitter_boxes": len(jitter_specs),
        "depth_model_kind": "depth_pro",
        "depth_model_semantics": "metric_model_depth_meters",
        "schema_compatible_with_source_aggregated_features_csv": True,
        "reused_columns": reusable_columns,
        "depth_columns_recomputed": depth_columns,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(
            "[attempt9/study03] Complete\n"
            "  aggregated_features_csv: {}\n"
            "  rows_written: {}\n"
            "  rows_skipped: {}\n"
            "  rows_excluded: {}".format(
                output_csv,
                len(completed_rows),
                len(skipped_rows),
                len(excluded_rows),
            )
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_feature_extraction(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
