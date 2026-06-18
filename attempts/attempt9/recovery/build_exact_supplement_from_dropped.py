"""
Build Attempt9 study01 exact-feature rows for study_all dropped exact rows.

The dropped rows already exist in the aggregated/jitter tables, but were missing
from the exact study01 table used by the unified benchmark. This script computes
only those missing exact rows with Depth Pro and writes a separate supplement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt9.recovery.build_problematic_folder_supplements import (  # noqa: E402
    _build_study01_supplement,
    _normalize_path,
    _reconstruct_sample_from_image,
    _write_manifest,
)
from pipeline.depth.depth_model import load_depth_pro  # noqa: E402


DEFAULT_DROPPED_AUDIT_CSV = (
    REPO_ROOT / "attempts/attempt9/study_all/artifacts_merged/audit/dropped_rows_audit.csv"
)
DEFAULT_EXISTING_EXACT_CSV = REPO_ROOT / "attempts/attempt9/merged/artifacts/study01/features_merged.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "attempts/attempt9/recovery/artifacts/study01_completion"
DEFAULT_OUTPUT_FILENAME = "features_study_all_dropped_exact.csv"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints/depth_pro.pt"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(rows: Iterable[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = _read_csv_rows(path)
    return {_normalize_path(row["image_path"]) for row in rows}


def _is_target_row(row: dict[str, str]) -> bool:
    reason = str(row.get("drop_reason", ""))
    return (
        "missing_exact_row" in reason
        and str(row.get("in_aggregated_rows", "")) == "1"
        and str(row.get("in_attempt4_split_universe", "")) == "1"
    )


def build_supplement(
    *,
    dropped_audit_csv: Path,
    existing_exact_csv: Path,
    output_root: Path,
    output_filename: str,
    checkpoint_path: Path,
    device: str | None,
    resume: bool,
    progress_every: int,
    verbose: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    dropped_rows = _read_csv_rows(dropped_audit_csv)
    existing_ids = _load_existing_ids(existing_exact_csv)
    target_rows = [row for row in dropped_rows if _is_target_row(row)]
    pending_rows = [
        row
        for row in target_rows
        if _normalize_path(row["source_group_id"]) not in existing_ids
    ]

    samples = []
    skipped_rows: list[dict[str, Any]] = []
    for row in pending_rows:
        image_path = Path(row["source_group_id"])
        try:
            samples.append(_reconstruct_sample_from_image(image_path))
        except (FileNotFoundError, ValueError, OSError) as exc:
            skipped_rows.append({"image_path": str(image_path), "reason": str(exc)})

    samples = sorted(samples, key=lambda sample: str(sample.image_path))
    _write_manifest(samples, audit_dir / "study01_completion_manifest.csv")
    _write_csv_rows(skipped_rows, audit_dir / "study01_completion_skipped.csv", ["image_path", "reason"])

    if verbose:
        print(
            "[attempt9/study01-completion] Setup\n"
            "  dropped_audit_csv: {}\n"
            "  existing_exact_csv: {}\n"
            "  output_root: {}\n"
            "  target_rows: {}\n"
            "  already_present_in_exact: {}\n"
            "  samples_to_compute: {}\n"
            "  skipped_before_compute: {}\n"
            "  checkpoint_path: {}\n"
            "  device: {}".format(
                dropped_audit_csv,
                existing_exact_csv,
                output_root,
                len(target_rows),
                len(target_rows) - len(pending_rows),
                len(samples),
                len(skipped_rows),
                checkpoint_path,
                device or "auto",
            )
        )

    depth_model = load_depth_pro(checkpoint_path=checkpoint_path, device=device)
    result = _build_study01_supplement(
        samples=samples,
        output_dir=output_root,
        output_filename=output_filename,
        depth_model=depth_model,
        resume=resume,
        progress_every=progress_every,
        verbose=verbose,
    )

    summary = {
        "study_name": "attempt9_study01_completion_from_study_all_dropped_rows",
        "dropped_audit_csv": str(dropped_audit_csv),
        "existing_exact_csv": str(existing_exact_csv),
        "output_root": str(output_root),
        "output_csv": result["output_csv"],
        "partial_csv": result["partial_csv"],
        "target_rows": len(target_rows),
        "already_present_in_exact": len(target_rows) - len(pending_rows),
        "samples_to_compute": len(samples),
        "skipped_before_compute": len(skipped_rows),
        "rows_written": result["rows"],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print("[attempt9/study01-completion] Complete\n  summary_json: {}".format(summary_path))
        print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropped-audit-csv", type=Path, default=DEFAULT_DROPPED_AUDIT_CSV)
    parser.add_argument("--existing-exact-csv", type=Path, default=DEFAULT_EXISTING_EXACT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-filename", default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build_supplement(
        dropped_audit_csv=_resolve_path(args.dropped_audit_csv),
        existing_exact_csv=_resolve_path(args.existing_exact_csv),
        output_root=_resolve_path(args.output_root),
        output_filename=args.output_filename,
        checkpoint_path=_resolve_path(args.checkpoint_path),
        device=args.device,
        resume=not args.no_resume,
        progress_every=max(1, int(args.progress_every)),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
