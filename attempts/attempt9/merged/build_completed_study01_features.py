"""
Create a completed Attempt9 study01 exact feature table.

It merges the existing clean+recovery study01 table with a small exact-feature
supplement generated for rows dropped by study_all because they were
aggregated-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_CSV = REPO_ROOT / "attempts/attempt9/merged/artifacts/study01/features_merged.csv"
DEFAULT_SUPPLEMENT_CSV = (
    REPO_ROOT
    / "attempts/attempt9/recovery/artifacts/study01_completion/features_study_all_dropped_exact.csv"
)
DEFAULT_OUTPUT_CSV = REPO_ROOT / "attempts/attempt9/merged/artifacts/study01/features_merged_completed.csv"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv_rows(rows: Iterable[dict[str, Any]], output_path: Path, fieldnames: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_image_id(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _sort_key(row: dict[str, Any]) -> tuple[float, str, str, str]:
    return (
        float(row["true_distance_m"]),
        str(row.get("weather", "")),
        str(row.get("time_of_day", "")),
        str(row["image_path"]),
    )


def build_completed_table(base_csv: Path, supplement_csv: Path, output_csv: Path) -> dict[str, Any]:
    base_rows, base_fieldnames = _read_csv_rows(base_csv)
    supplement_rows, supplement_fieldnames = _read_csv_rows(supplement_csv)
    if base_fieldnames != supplement_fieldnames:
        raise ValueError(
            "Schema mismatch between base and supplement CSVs.\n"
            "base: {}\n"
            "supplement: {}".format(base_fieldnames, supplement_fieldnames)
        )

    base_ids = {_normalize_image_id(row["image_path"]) for row in base_rows}
    duplicate_supplement_rows = [
        row["image_path"]
        for row in supplement_rows
        if _normalize_image_id(row["image_path"]) in base_ids
    ]
    if duplicate_supplement_rows:
        raise ValueError(
            "Supplement contains rows already present in base. First duplicates: {}".format(
                duplicate_supplement_rows[:10]
            )
        )

    completed_rows = sorted(list(base_rows) + list(supplement_rows), key=_sort_key)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in completed_rows:
        image_id = _normalize_image_id(row["image_path"])
        if image_id in seen:
            duplicates.append(row["image_path"])
        seen.add(image_id)
    if duplicates:
        raise ValueError("Completed table has duplicate image rows: {}".format(duplicates[:10]))

    _write_csv_rows(completed_rows, output_csv, base_fieldnames)
    summary = {
        "base_csv": str(base_csv),
        "supplement_csv": str(supplement_csv),
        "output_csv": str(output_csv),
        "base_rows": len(base_rows),
        "supplement_rows": len(supplement_rows),
        "completed_rows": len(completed_rows),
        "column_count": len(base_fieldnames),
        "schema_compatible": True,
        "duplicate_rows": 0,
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--supplement-csv", type=Path, default=DEFAULT_SUPPLEMENT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    build_completed_table(
        base_csv=_resolve_path(args.base_csv),
        supplement_csv=_resolve_path(args.supplement_csv),
        output_csv=_resolve_path(args.output_csv),
    )


if __name__ == "__main__":
    main()
