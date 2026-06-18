"""
Build an Attempt2/Study06-compatible fused depth feature table with Depth Pro.

The output keeps the same wide-table schema used by Attempt8, but recomputes
the depth-dependent feature columns from Apple's Depth Pro model. All metadata
columns are reused from the original Study06 fused table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.data.loader import (  # noqa: E402
    DroneDatasetSample,
    _read_image_size,
    extract_metadata_from_filename,
    load_image_rgb,
    parse_yolo_label_file,
)
from pipeline.depth.depth_model import estimate_depth_map, load_depth_pro  # noqa: E402
from pipeline.depth.representation import (  # noqa: E402
    compute_depth_representation,
    extract_context_crop,
    resolve_context_window_specs,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_SOURCE_FUSED_CSV = (
    REPO_ROOT
    / "attempts/attempt2/studies/study_06_all_images_midpoint_random_forest/artifacts/fused_features/depth_only_feature_table.csv"
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


def _method_id(context_name: str, aggregation_method: str, score_field: str) -> str:
    return "{}__{}__{}".format(context_name, aggregation_method, score_field)


def _depth_feature_names(fieldnames: list[str]) -> list[str]:
    return [
        name
        for name in fieldnames
        if "__object_depth" in name
        or "__object_depth_percentile_5_95" in name
        or "__object_minus_ring" in name
        or "__object_minus_ring_normalized" in name
        or "__surrounding_ring_depth" in name
    ]


def _load_partial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {_normalize_image_id(row["image_path"]) for row in _read_csv_rows(path)}


def _load_partial_rows(path: Path, excluded_path_fragments: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [
        row
        for row in _read_csv_rows(path)
        if not _is_excluded_path(row["image_path"], excluded_path_fragments)
    ]


def _compute_depth_features(
    row: dict[str, Any],
    depth_model: Any,
    context_specs: list[Any],
    aggregation_methods: list[str],
    score_fields: list[str],
    inner_bbox_scale: float,
    surrounding_bbox_scale: float,
) -> dict[str, float]:
    sample = _reconstruct_sample(row)
    image = load_image_rgb(sample.image_path)
    computed: dict[str, float] = {}
    for context_spec in context_specs:
        crop_data = extract_context_crop(
            image=image,
            bbox=sample.annotation.bbox,
            context_spec=context_spec,
        )
        depth_map = estimate_depth_map(crop_data["image"], depth_model)
        for aggregation_method in aggregation_methods:
            representation = compute_depth_representation(
                depth_map=depth_map,
                bbox=crop_data["local_bbox"],
                aggregation_method=aggregation_method,
                inner_bbox_scale=inner_bbox_scale,
                surrounding_bbox_scale=surrounding_bbox_scale,
            )
            for score_field in score_fields:
                computed[_method_id(context_spec.name, aggregation_method, score_field)] = round(
                    float(representation[score_field]),
                    8,
                )
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
        REPO_ROOT / "attempts/attempt9/study02/artifacts",
    )
    features_dir = output_root / "fused_features"
    audit_dir = output_root / "audit"
    features_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    source_csv = _resolve_repo_path(
        dict(resolved_config.get("sources", {})).get("source_fused_features_csv"),
        DEFAULT_SOURCE_FUSED_CSV,
    )
    source_rows = _read_csv_rows(source_csv)
    max_samples = resolved_config.get("max_samples")
    if max_samples is not None:
        source_rows = source_rows[: int(max_samples)]

    with source_csv.open("r", newline="", encoding="utf-8") as handle:
        source_fieldnames = list(csv.DictReader(handle).fieldnames or [])
    depth_columns = _depth_feature_names(source_fieldnames)

    study_config = dict(resolved_config.get("study", {}))
    context_specs = list(resolve_context_window_specs(study_config.get("context_windows")))
    aggregation_methods = list(study_config.get("aggregation_methods", []))
    score_fields = list(study_config.get("score_fields", ["object_depth"]))
    progress_every = max(1, int(study_config.get("progress_log_every_rows", 25)))
    resume_from_partial = bool(study_config.get("resume_from_partial", True))
    stop_on_missing_source = bool(study_config.get("stop_on_missing_source", False))
    excluded_path_fragments = _resolve_excluded_path_fragments(study_config)

    output_csv = features_dir / "depth_only_feature_table.csv"
    partial_csv = features_dir / "depth_only_feature_table_partial.csv"
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
            "[attempt9/study02] Building Study06 fused table with Depth Pro\n"
            "  source_csv: {}\n"
            "  output_csv: {}\n"
            "  partial_csv: {}\n"
            "  source_rows: {}\n"
            "  reusable_non_depth_columns: {}\n"
            "  depth_columns_to_recompute: {}\n"
            "  excluded_path_fragments: {}".format(
                source_csv,
                output_csv,
                partial_csv,
                total,
                len(source_fieldnames) - len(depth_columns),
                len(depth_columns),
                excluded_path_fragments,
            )
        )
        if len(completed_rows) != len(partial_rows):
            print(
                "[attempt9/study02] Removed excluded rows from partial cache\n"
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
                score_fields=score_fields,
                inner_bbox_scale=float(study_config.get("inner_bbox_scale", 0.5)),
                surrounding_bbox_scale=float(study_config.get("surrounding_bbox_scale", 1.8)),
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
                    "[attempt9/study02] Skipping source row {}/{}\n"
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
                "[attempt9/study02] Row {}/{} | written={} | skipped={} | elapsed={} | eta={}".format(
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
        "source_fused_features_csv": str(source_csv),
        "fused_features_csv": str(output_csv),
        "partial_fused_features_csv": str(partial_csv),
        "skipped_source_rows_csv": str(skipped_csv),
        "source_rows": total,
        "rows_written": len(completed_rows),
        "rows_skipped": len(skipped_rows),
        "rows_excluded": len(excluded_rows),
        "excluded_path_fragments": excluded_path_fragments,
        "depth_model_kind": "depth_pro",
        "depth_model_semantics": "metric_model_depth_meters",
        "schema_compatible_with_source_fused_features_csv": True,
        "depth_columns_recomputed": depth_columns,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(
            "[attempt9/study02] Complete\n"
            "  fused_features_csv: {}\n"
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
