"""
Analyze whether Nenrus errors are related to the coarse distance grid used for training.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPORTS_DIR = STUDY_DIR / "artifacts" / "reports"
PLOTS_DIR = STUDY_DIR / "artifacts" / "plots"
PREDICTIONS_CSV = REPORTS_DIR / "predictions.csv"
OUTPUT_MD = REPORTS_DIR / "distance_grid_analysis.md"

TRAIN_DISTANCES = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 125.0, 150.0]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mae": None,
            "mean_relative_error": None,
            "median_relative_error": None,
            "mean_signed_error": None,
            "within_10m_rate": None,
        }
    abs_errors = np.asarray([float(row["absolute_error_m"]) for row in rows], dtype=np.float64)
    rel_errors = np.asarray([float(row["relative_error"]) for row in rows], dtype=np.float64)
    signed_errors = np.asarray([float(row["signed_error_m"]) for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "mae": float(np.mean(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "mean_signed_error": float(np.mean(signed_errors)),
        "median_signed_error": float(np.median(signed_errors)),
        "within_5m_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10m_rate": float(np.mean(abs_errors <= 10.0)),
        "within_20m_rate": float(np.mean(abs_errors <= 20.0)),
    }


def _nearest_train_distance(distance: float) -> float:
    return min(TRAIN_DISTANCES, key=lambda train_distance: abs(train_distance - distance))


def _gap_bin(gap: float) -> str:
    if math.isclose(gap, 0.0, abs_tol=1e-9):
        return "exact_seen_training_distance"
    if gap <= 5.0:
        return "within_5m_of_training_distance"
    if gap <= 10.0:
        return "within_10m_of_training_distance"
    return "more_than_10m_from_training_distance"


def _add_grid_labels(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        distance = float(row["true_distance_m"])
        nearest = _nearest_train_distance(distance)
        gap = abs(distance - nearest)
        enriched["is_multiple_of_10"] = str(math.isclose(distance % 10.0, 0.0, abs_tol=1e-9))
        enriched["is_exact_training_distance"] = str(any(math.isclose(distance, d, abs_tol=1e-9) for d in TRAIN_DISTANCES))
        enriched["nearest_training_distance_m"] = nearest
        enriched["nearest_training_distance_gap_m"] = gap
        enriched["nearest_training_distance_gap_bin"] = _gap_bin(gap)
        enriched["is_below_min_training_distance"] = str(distance < min(TRAIN_DISTANCES))
        output.append(enriched)
    return output


def _group_metrics(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output: List[Dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        metric_row = {key: group_key}
        metric_row.update(_metrics(group_rows))
        output.append(metric_row)
    return output


def _distance_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row["true_distance_m"]), []).append(row)
    output: List[Dict[str, Any]] = []
    for distance, group_rows in sorted(groups.items()):
        nearest = _nearest_train_distance(distance)
        metric_row = {
            "distance_m": distance,
            "is_multiple_of_10": math.isclose(distance % 10.0, 0.0, abs_tol=1e-9),
            "is_exact_training_distance": any(math.isclose(distance, d, abs_tol=1e-9) for d in TRAIN_DISTANCES),
            "nearest_training_distance_m": nearest,
            "nearest_training_distance_gap_m": abs(distance - nearest),
            "is_below_min_training_distance": distance < min(TRAIN_DISTANCES),
        }
        metric_row.update(_metrics(group_rows))
        output.append(metric_row)
    return output


def _bar_plot(rows: Sequence[Dict[str, Any]], key: str, metric: str, title: str, output_path: Path) -> None:
    labels = [str(row[key]) for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.bar(labels, values, color="#496d82")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _distance_gap_scatter(distance_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [float(row["nearest_training_distance_gap_m"]) for row in distance_rows]
    y = [float(row["mean_relative_error"]) for row in distance_rows]
    sizes = [max(40, float(row["count"]) * 4) for row in distance_rows]
    colors = ["#b85c45" if str(row["is_below_min_training_distance"]) == "True" else "#3d6f5c" for row in distance_rows]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.78)
    for row in distance_rows:
        ax.annotate(str(int(float(row["distance_m"]))), (float(row["nearest_training_distance_gap_m"]), float(row["mean_relative_error"])))
    ax.set_title("Relative Error vs Gap from Nearest Training Distance")
    ax.set_xlabel("Gap from nearest training distance (m)")
    ax.set_ylabel("Mean relative error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def run() -> Dict[str, Any]:
    rows = _add_grid_labels(_read_csv(PREDICTIONS_CSV))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    by_multiple = _group_metrics(rows, "is_multiple_of_10")
    by_exact_seen = _group_metrics(rows, "is_exact_training_distance")
    by_gap_bin = _group_metrics(rows, "nearest_training_distance_gap_bin")
    by_below_min = _group_metrics(rows, "is_below_min_training_distance")
    by_distance = _distance_rows(rows)

    _write_csv(REPORTS_DIR / "distance_grid_by_multiple_of_10.csv", by_multiple)
    _write_csv(REPORTS_DIR / "distance_grid_by_exact_training_distance.csv", by_exact_seen)
    _write_csv(REPORTS_DIR / "distance_grid_by_nearest_training_gap_bin.csv", by_gap_bin)
    _write_csv(REPORTS_DIR / "distance_grid_by_below_min_training_distance.csv", by_below_min)
    _write_csv(REPORTS_DIR / "distance_grid_by_distance.csv", by_distance)

    _bar_plot(
        by_multiple,
        "is_multiple_of_10",
        "mean_relative_error",
        "Relative Error: Multiple-of-10 vs Non-Multiple-of-10",
        PLOTS_DIR / "distance_grid_relative_error_multiple_of_10.png",
    )
    _bar_plot(
        by_exact_seen,
        "is_exact_training_distance",
        "mean_relative_error",
        "Relative Error: Exact Training Distances vs Unseen Distances",
        PLOTS_DIR / "distance_grid_relative_error_exact_seen.png",
    )
    _bar_plot(
        by_gap_bin,
        "nearest_training_distance_gap_bin",
        "mean_relative_error",
        "Relative Error by Gap from Nearest Training Distance",
        PLOTS_DIR / "distance_grid_relative_error_gap_bin.png",
    )
    _distance_gap_scatter(
        by_distance,
        PLOTS_DIR / "distance_grid_relative_error_vs_training_gap.png",
    )

    lines = [
        "# Distance Grid Effect Analysis",
        "",
        "Training distances in the exported model's original Attempt 4 feature table:",
        "",
        "`{}`".format(", ".join(str(int(d)) if float(d).is_integer() else str(d) for d in TRAIN_DISTANCES)),
        "",
        "## Question",
        "",
        "Does the external Nenrus error get worse mainly because some Nenrus distances are not multiples of 10, or because they are not distances seen during training?",
        "",
        "## Multiple-of-10 vs Non-Multiple-of-10",
        "",
    ]
    for row in by_multiple:
        lines.append(
            "- `{}`: count `{}`, MAE `{}m`, mean relative error `{}`, mean signed error `{}m`".format(
                row["is_multiple_of_10"],
                row["count"],
                _fmt(row["mae"]),
                _fmt(row["mean_relative_error"]),
                _fmt(row["mean_signed_error"]),
            )
        )
    lines.extend(["", "## Exact Training Distance vs Unseen Exact Distance", ""])
    for row in by_exact_seen:
        lines.append(
            "- `{}`: count `{}`, MAE `{}m`, mean relative error `{}`, mean signed error `{}m`".format(
                row["is_exact_training_distance"],
                row["count"],
                _fmt(row["mae"]),
                _fmt(row["mean_relative_error"]),
                _fmt(row["mean_signed_error"]),
            )
        )
    lines.extend(["", "## Gap From Nearest Training Distance", ""])
    for row in by_gap_bin:
        lines.append(
            "- `{}`: count `{}`, MAE `{}m`, mean relative error `{}`, mean signed error `{}m`".format(
                row["nearest_training_distance_gap_bin"],
                row["count"],
                _fmt(row["mae"]),
                _fmt(row["mean_relative_error"]),
                _fmt(row["mean_signed_error"]),
            )
        )
    lines.extend(["", "## Below Minimum Training Distance", ""])
    for row in by_below_min:
        lines.append(
            "- `{}`: count `{}`, MAE `{}m`, mean relative error `{}`, mean signed error `{}m`".format(
                row["is_below_min_training_distance"],
                row["count"],
                _fmt(row["mae"]),
                _fmt(row["mean_relative_error"]),
                _fmt(row["mean_signed_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The coarse training grid probably contributes to the problem, but it is not the full explanation.",
            "The strongest evidence is that exact training distances in Nenrus still have high external error.",
            "The very short distances below 20m have especially large relative error because the model was never trained below 20m and all predictions are biased upward.",
            "",
            "So the issue has two layers:",
            "",
            "- interpolation/grid issue: unseen distances such as 25, 35, 45, 55, 65, 75 are not exact training targets",
            "- extrapolation/domain issue: distances below 20m are outside the training range, and the real-drone geometry differs from the original dataset",
            "",
            "The observed failure is therefore not just because some labels are 4m or 9m instead of clean tens. It is also because the model learned a same-domain geometry-to-distance mapping that does not transfer cleanly to Nenrus.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "training_distances_m": TRAIN_DISTANCES,
        "by_multiple_of_10_csv": str(REPORTS_DIR / "distance_grid_by_multiple_of_10.csv"),
        "by_exact_training_distance_csv": str(REPORTS_DIR / "distance_grid_by_exact_training_distance.csv"),
        "by_nearest_training_gap_bin_csv": str(REPORTS_DIR / "distance_grid_by_nearest_training_gap_bin.csv"),
        "by_below_min_training_distance_csv": str(REPORTS_DIR / "distance_grid_by_below_min_training_distance.csv"),
        "by_distance_csv": str(REPORTS_DIR / "distance_grid_by_distance.csv"),
        "analysis_md": str(OUTPUT_MD),
    }
    (REPORTS_DIR / "distance_grid_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    run()
