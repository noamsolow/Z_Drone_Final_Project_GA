"""
Build mini study 05 tables for raw relative depth values from Depth Anything.

For each aggregation method, this script computes:
- rows: exact ground-truth distances
- columns: zoom contexts
- cell: mean raw relative depth (`object_depth`)

Outputs:
- artifacts/reports/relative_depth_tables.md
- artifacts/reports/relative_depth_table_middle.csv
- artifacts/reports/relative_depth_table_mean.csv
- artifacts/reports/relative_depth_table_median.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    STUDY_DIR.parent
    / "study_04_depth_only_models_30_per_stratum"
    / "artifacts"
    / "features"
    / "representation_records.csv"
)
DEFAULT_OUTPUT_DIR = STUDY_DIR / "artifacts" / "reports"

AGGREGATION_SPECS: List[Tuple[str, str]] = [
    ("bbox_midpoint", "middle"),
    ("bbox_mean", "mean"),
    ("inner50_median", "median"),
]

AGGREGATION_LABELS: Dict[str, str] = {
    "bbox_midpoint": "middle",
    "bbox_mean": "mean",
    "inner50_median": "median",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mini study 05 relative-depth tables from cached feature rows."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the representation_records.csv file to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the markdown and CSV tables will be written.",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _zoom_sort_key(context_name: str, context_scale: str) -> Tuple[float, str]:
    if context_name == "full_image" or context_scale == "full_image":
        return (float("inf"), context_name)
    return (float(context_scale), context_name)


def _zoom_label(context_name: str, context_scale: str) -> str:
    if context_name == "full_image" or context_scale == "full_image":
        return "full"
    scale = float(context_scale)
    if scale.is_integer():
        return "{}x".format(int(scale))
    return "{}x".format(context_scale)


def _distance_label(distance_value: float) -> str:
    if distance_value.is_integer():
        return str(int(distance_value))
    return "{:.2f}".format(distance_value).rstrip("0").rstrip(".")


def _collect_orders(rows: Iterable[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str], List[float]]:
    context_to_scale: Dict[str, str] = {}
    distances = set()
    for row in rows:
        context_to_scale[str(row["context_window"])] = str(row["context_scale"])
        distances.add(float(row["true_distance_m"]))

    ordered_contexts = sorted(
        context_to_scale,
        key=lambda context_name: _zoom_sort_key(context_name, context_to_scale[context_name]),
    )
    ordered_distances = sorted(distances)
    return ordered_contexts, context_to_scale, ordered_distances


def _compute_mean_tables(
    rows: Iterable[Dict[str, Any]],
    ordered_contexts: List[str],
    ordered_distances: List[float],
) -> Dict[str, Dict[float, Dict[str, float]]]:
    sums: Dict[str, Dict[float, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: Dict[str, Dict[float, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in rows:
        aggregation_method = str(row["aggregation_method"])
        if aggregation_method not in AGGREGATION_LABELS:
            continue
        distance = float(row["true_distance_m"])
        context_name = str(row["context_window"])
        object_depth = float(row["object_depth"])
        sums[aggregation_method][distance][context_name] += object_depth
        counts[aggregation_method][distance][context_name] += 1

    tables: Dict[str, Dict[float, Dict[str, float]]] = {}
    for aggregation_method, _ in AGGREGATION_SPECS:
        table: Dict[float, Dict[str, float]] = {}
        for distance in ordered_distances:
            row_values: Dict[str, float] = {}
            for context_name in ordered_contexts:
                count = counts[aggregation_method][distance][context_name]
                if count <= 0:
                    row_values[context_name] = float("nan")
                else:
                    row_values[context_name] = sums[aggregation_method][distance][context_name] / count
            table[distance] = row_values
        tables[aggregation_method] = table
    return tables


def _write_csv_table(
    output_path: Path,
    aggregation_method: str,
    table: Dict[float, Dict[str, float]],
    ordered_contexts: List[str],
    context_to_scale: Dict[str, str],
    ordered_distances: List[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["true_distance_m"] + [
        _zoom_label(context_name, context_to_scale[context_name]) for context_name in ordered_contexts
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for distance in ordered_distances:
            row = [_distance_label(distance)]
            for context_name in ordered_contexts:
                value = table[distance][context_name]
                row.append("{:.6f}".format(value))
            writer.writerow(row)


def _markdown_table(
    aggregation_method: str,
    table: Dict[float, Dict[str, float]],
    ordered_contexts: List[str],
    context_to_scale: Dict[str, str],
    ordered_distances: List[float],
) -> str:
    header_cells = ["true_depth_m"] + [
        _zoom_label(context_name, context_to_scale[context_name]) for context_name in ordered_contexts
    ]
    separator = ["---"] * len(header_cells)
    lines = [
        "## {} table".format(AGGREGATION_LABELS[aggregation_method].capitalize()),
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for distance in ordered_distances:
        cells = [_distance_label(distance)]
        for context_name in ordered_contexts:
            cells.append("{:.4f}".format(table[distance][context_name]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_markdown_report(
    output_path: Path,
    input_csv: Path,
    ordered_contexts: List[str],
    context_to_scale: Dict[str, str],
    ordered_distances: List[float],
    tables: Dict[str, Dict[float, Dict[str, float]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zoom_summary = ", ".join(
        _zoom_label(context_name, context_to_scale[context_name]) for context_name in ordered_contexts
    )
    distance_summary = ", ".join(_distance_label(distance) for distance in ordered_distances)
    sections = [
        "# Mini Study 05: Relative Depth Tables",
        "",
        "Source CSV: `{}`".format(input_csv.as_posix()),
        "",
        "Each cell is the mean `object_depth` value from Depth Anything for one exact `(true depth, zoom)` pair.",
        "",
        "Zoom columns: {}".format(zoom_summary),
        "",
        "Ground-truth rows: {} m".format(distance_summary),
        "",
    ]
    for aggregation_method, _ in AGGREGATION_SPECS:
        sections.append(
            _markdown_table(
                aggregation_method=aggregation_method,
                table=tables[aggregation_method],
                ordered_contexts=ordered_contexts,
                context_to_scale=context_to_scale,
                ordered_distances=ordered_distances,
            )
        )
    output_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = _read_csv_rows(args.input_csv)
    ordered_contexts, context_to_scale, ordered_distances = _collect_orders(rows)
    tables = _compute_mean_tables(rows, ordered_contexts, ordered_distances)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for aggregation_method, output_name in AGGREGATION_SPECS:
        _write_csv_table(
            output_path=args.output_dir / "relative_depth_table_{}.csv".format(output_name),
            aggregation_method=aggregation_method,
            table=tables[aggregation_method],
            ordered_contexts=ordered_contexts,
            context_to_scale=context_to_scale,
            ordered_distances=ordered_distances,
        )

    _write_markdown_report(
        output_path=args.output_dir / "relative_depth_tables.md",
        input_csv=args.input_csv,
        ordered_contexts=ordered_contexts,
        context_to_scale=context_to_scale,
        ordered_distances=ordered_distances,
        tables=tables,
    )

    print("Mini Study 05 tables written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
