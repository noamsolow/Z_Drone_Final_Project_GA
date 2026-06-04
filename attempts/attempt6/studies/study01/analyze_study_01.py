"""
Generate focused analysis plots for attempt6 Study 01.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPORTS_DIR = STUDY_DIR / "artifacts" / "reports"
PLOTS_DIR = STUDY_DIR / "artifacts" / "plots"

SUMMARY_PATH = REPORTS_DIR / "summary.json"
REAL_BY_DISTANCE_PATH = REPORTS_DIR / "real_by_distance.csv"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _metric_rows_by_drone_distance(
    rows: Sequence[Dict[str, Any]],
    winner_variant_name: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["variant_name"]) != winner_variant_name:
            continue
        if str(row["split_name"]) != "real_external" or str(row["model_name"]) != "blend":
            continue
        grouped[(str(row["drone_type"]), float(row["true_distance_m"]))].append(row)

    output: List[Dict[str, Any]] = []
    for (drone_type, distance_m), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        relative_errors = np.asarray([float(row["relative_error"]) for row in group_rows], dtype=np.float64)
        absolute_errors = np.asarray([float(row["absolute_error_m"]) for row in group_rows], dtype=np.float64)
        output.append(
            {
                "variant_name": winner_variant_name,
                "drone_type": drone_type,
                "true_distance_m": distance_m,
                "count": int(relative_errors.size),
                "mae": float(np.mean(absolute_errors)),
                "mean_relative_error": float(np.mean(relative_errors)),
                "mean_relative_error_percent": float(100.0 * np.mean(relative_errors)),
            }
        )
    return output


def _plot_all_variants_relative_error(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    filtered = [
        row for row in rows
        if str(row["split_name"]) == "real_external" and str(row["model_name"]) == "blend"
    ]
    variants = sorted({str(row["variant_name"]) for row in filtered})
    fig, ax = plt.subplots(figsize=(12, 7))
    for variant_name in variants:
        variant_rows = sorted(
            [row for row in filtered if str(row["variant_name"]) == variant_name],
            key=lambda row: float(row["true_distance_m"]),
        )
        distances = [float(row["true_distance_m"]) for row in variant_rows]
        values = [100.0 * float(row["mean_relative_error"]) for row in variant_rows]
        line_width = 3.0 if variant_name == "reduced_geometry" else 1.7
        alpha = 1.0 if variant_name == "reduced_geometry" else 0.65
        ax.plot(distances, values, marker="o", linewidth=line_width, alpha=alpha, label=variant_name)

    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("Attempt6 Study01 Real External Relative Error by Distance")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_winner_relative_error(rows: Sequence[Dict[str, Any]], winner_variant_name: str, output_path: Path) -> None:
    winner_rows = sorted(
        [
            row for row in rows
            if str(row["split_name"]) == "real_external"
            and str(row["model_name"]) == "blend"
            and str(row["variant_name"]) == winner_variant_name
        ],
        key=lambda row: float(row["true_distance_m"]),
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    distances = [float(row["true_distance_m"]) for row in winner_rows]
    values = [100.0 * float(row["mean_relative_error"]) for row in winner_rows]
    ax.bar(distances, values, width=2.6, color="#315f8c")
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("Winner Relative Error by Distance ({})".format(winner_variant_name))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_winner_by_drone(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    drone_types = sorted({str(row["drone_type"]) for row in rows})
    colors = {"Kongsberg": "#b85c45", "Vestfold": "#315f8c"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for drone_type in drone_types:
        drone_rows = sorted(
            [row for row in rows if str(row["drone_type"]) == drone_type],
            key=lambda row: float(row["true_distance_m"]),
        )
        distances = [float(row["true_distance_m"]) for row in drone_rows]
        values = [float(row["mean_relative_error_percent"]) for row in drone_rows]
        ax.plot(distances, values, marker="o", linewidth=2.2, label=drone_type, color=colors.get(drone_type, "#666666"))
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("Winner Relative Error by Distance and Drone Type")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = _read_json(SUMMARY_PATH)
    winner_variant_name = str(summary["winner_by_real_blend_mae"]["variant_name"])
    real_by_distance_rows = _read_csv_rows(REAL_BY_DISTANCE_PATH)
    prediction_rows = _read_csv_rows(PREDICTIONS_PATH)

    winner_by_drone_distance_rows = _metric_rows_by_drone_distance(prediction_rows, winner_variant_name)
    _write_csv_rows(REPORTS_DIR / "winner_real_relative_error_by_drone_distance.csv", winner_by_drone_distance_rows)

    _plot_all_variants_relative_error(
        real_by_distance_rows,
        PLOTS_DIR / "real_relative_error_by_distance_all_variants.png",
    )
    _plot_winner_relative_error(
        real_by_distance_rows,
        winner_variant_name,
        PLOTS_DIR / "winner_real_relative_error_by_distance.png",
    )
    _plot_winner_by_drone(
        winner_by_drone_distance_rows,
        PLOTS_DIR / "winner_real_relative_error_by_distance_and_drone_type.png",
    )

    print("[attempt6-study01] Analysis plots written to {}".format(PLOTS_DIR))


if __name__ == "__main__":
    main()
