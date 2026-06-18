"""
Merge Attempt9 clean datasets with the separately computed recovery supplements.

The script writes new merged CSVs and never overwrites the clean-only or
recovery-only feature tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _resolve_repo_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or REPO_ROOT) / path


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv_rows(rows: Iterable[dict[str, Any]], output_path: Path, fieldnames: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_image_id(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _sort_key(row: dict[str, Any]) -> tuple[float, str, str, str]:
    return (
        float(row["true_distance_m"]),
        str(row.get("weather", "")),
        str(row.get("time_of_day", "")),
        str(row["image_path"]),
    )


def _merge_pair(
    clean_csv: Path,
    recovery_csv: Path,
    output_csv: Path,
    dataset_name: str,
) -> dict[str, Any]:
    clean_rows, clean_fieldnames = _read_csv_rows(clean_csv)
    recovery_rows, recovery_fieldnames = _read_csv_rows(recovery_csv)
    if clean_fieldnames != recovery_fieldnames:
        raise ValueError(
            "{} schema mismatch between clean and recovery CSVs.".format(dataset_name)
        )

    merged_rows = list(clean_rows) + list(recovery_rows)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in merged_rows:
        image_id = _normalize_image_id(row["image_path"])
        if image_id in seen:
            duplicates.append(row["image_path"])
        seen.add(image_id)
    if duplicates:
        raise ValueError(
            "{} merge would create duplicate image_path rows. First duplicates: {}".format(
                dataset_name,
                duplicates[:10],
            )
        )

    merged_rows = sorted(merged_rows, key=_sort_key)
    _write_csv_rows(merged_rows, output_csv, clean_fieldnames)
    return {
        "dataset": dataset_name,
        "clean_csv": str(clean_csv),
        "recovery_csv": str(recovery_csv),
        "merged_csv": str(output_csv),
        "clean_rows": len(clean_rows),
        "recovery_rows": len(recovery_rows),
        "merged_rows": len(merged_rows),
        "column_count": len(clean_fieldnames),
        "schema_compatible": True,
    }


def _load_image_ids(path: Path) -> set[str]:
    rows, _ = _read_csv_rows(path)
    return {_normalize_image_id(row["image_path"]) for row in rows}


def run_merge(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = _read_yaml(config_path)
    sources = dict(config.get("sources", {}))
    outputs = dict(config.get("outputs", {}))
    output_root = _resolve_repo_path(config.get("output_root", "attempts/attempt9/merged/artifacts"))
    output_root.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "study01",
            _resolve_repo_path(sources["study01_clean_csv"]),
            _resolve_repo_path(sources["study01_recovery_csv"]),
            _resolve_repo_path(outputs["study01_merged_csv"], base=output_root),
        ),
        (
            "study02",
            _resolve_repo_path(sources["study02_clean_csv"]),
            _resolve_repo_path(sources["study02_recovery_csv"]),
            _resolve_repo_path(outputs["study02_merged_csv"], base=output_root),
        ),
        (
            "study03",
            _resolve_repo_path(sources["study03_clean_csv"]),
            _resolve_repo_path(sources["study03_recovery_csv"]),
            _resolve_repo_path(outputs["study03_merged_csv"], base=output_root),
        ),
    ]

    results = [
        _merge_pair(
            clean_csv=clean_csv,
            recovery_csv=recovery_csv,
            output_csv=output_csv,
            dataset_name=dataset_name,
        )
        for dataset_name, clean_csv, recovery_csv, output_csv in specs
    ]

    id_sets = {
        result["dataset"]: _load_image_ids(Path(result["merged_csv"]))
        for result in results
    }
    common_ids = set.intersection(*id_sets.values())
    summary = {
        "study_name": config.get("study_name", "attempt9_merged_clean_plus_recovery"),
        "output_root": str(output_root),
        "datasets": results,
        "common_image_ids_across_merged_sources": len(common_ids),
        "study01_missing_from_study02": len(id_sets["study01"] - id_sets["study02"]),
        "study02_missing_from_study01": len(id_sets["study02"] - id_sets["study01"]),
        "study01_missing_from_study03": len(id_sets["study01"] - id_sets["study03"]),
        "study03_missing_from_study01": len(id_sets["study03"] - id_sets["study01"]),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    run_merge(config_path=args.config)


if __name__ == "__main__":
    main()
