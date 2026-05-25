"""
Generate a full visual analysis package for Study 03.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

SCORE_FIELD_LABELS = {
    "object_depth": "Raw object depth",
    "object_depth_percentile_5_95": "Object depth percentile (5-95)",
    "object_minus_ring": "Object minus ring",
    "object_minus_ring_normalized": "Normalized object minus ring",
}

AGGREGATION_LABELS = {
    "bbox_midpoint": "BBox midpoint",
    "bbox_mean": "BBox mean",
    "inner50_median": "Inner-50 median",
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv_rows(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _context_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(config.get("study", {}).get("context_windows", []))


def _context_order_and_labels(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    context_specs = _context_specs(config)
    ordered_names: List[str] = []
    labels: List[str] = []
    for spec in context_specs:
        name = str(spec["name"])
        scale = spec.get("scale")
        ordered_names.append(name)
        if scale is None:
            labels.append("full")
        elif float(scale).is_integer():
            labels.append("{}x".format(int(float(scale))))
        else:
            labels.append("{}x".format(scale))
    return ordered_names, labels


def _to_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def _to_int(row: Dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _sort_single_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: int(float(row["rank"])))


def _plot_best_by_context(
    output_path: Path,
    ordered_contexts: Sequence[str],
    context_labels: Sequence[str],
    single_rows: Sequence[Dict[str, Any]],
) -> None:
    best_by_context: Dict[str, Dict[str, Any]] = {}
    for row in single_rows:
        context = str(row["context_window"])
        if context not in best_by_context:
            best_by_context[context] = row

    maes = [_to_float(best_by_context[context], "loocv_mae") for context in ordered_contexts]
    best_methods = [best_by_context[context]["method_id"] for context in ordered_contexts]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(context_labels, maes, color="#2f6db3")
    ax.set_title("Best Single-Scale MAE by Zoom Context")
    ax.set_xlabel("Context window")
    ax.set_ylabel("LOOCV MAE (m)")
    ax.grid(axis="y", alpha=0.25)
    for bar, mae, method_id in zip(bars, maes, best_methods):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.2,
            "{:.2f}\n{}".format(mae, method_id.split("__")[1]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_score_field_zoom_trends(
    output_path: Path,
    ordered_contexts: Sequence[str],
    context_labels: Sequence[str],
    single_rows: Sequence[Dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    score_fields = [
        "object_depth",
        "object_depth_percentile_5_95",
        "object_minus_ring",
        "object_minus_ring_normalized",
    ]
    aggregation_methods = ["bbox_midpoint", "bbox_mean", "inner50_median"]
    colors = {
        "bbox_midpoint": "#1f77b4",
        "bbox_mean": "#ff7f0e",
        "inner50_median": "#2ca02c",
    }

    row_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in single_rows:
        row_lookup[(str(row["context_window"]), str(row["aggregation_method"]), str(row["score_field"]))] = row

    for axis, score_field in zip(axes, score_fields):
        for aggregation_method in aggregation_methods:
            y_values = []
            for context in ordered_contexts:
                row = row_lookup[(context, aggregation_method, score_field)]
                y_values.append(_to_float(row, "loocv_mae"))
            axis.plot(
                context_labels,
                y_values,
                marker="o",
                linewidth=2,
                color=colors[aggregation_method],
                label=AGGREGATION_LABELS[aggregation_method],
            )
        axis.set_title(SCORE_FIELD_LABELS[score_field])
        axis.set_ylabel("LOOCV MAE (m)")
        axis.grid(alpha=0.25)
    axes[2].set_xlabel("Context window")
    axes[3].set_xlabel("Context window")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("How Each Score Field Changes with Zoom and Aggregation", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_context_score_heatmap(
    output_path: Path,
    ordered_contexts: Sequence[str],
    context_labels: Sequence[str],
    single_rows: Sequence[Dict[str, Any]],
) -> None:
    score_fields = [
        "object_depth",
        "object_depth_percentile_5_95",
        "object_minus_ring",
        "object_minus_ring_normalized",
    ]
    matrix = np.zeros((len(score_fields), len(ordered_contexts)), dtype=np.float64)
    for score_index, score_field in enumerate(score_fields):
        for context_index, context in enumerate(ordered_contexts):
            candidate_rows = [
                row
                for row in single_rows
                if str(row["context_window"]) == context and str(row["score_field"]) == score_field
            ]
            best_row = min(candidate_rows, key=lambda row: _to_float(row, "loocv_mae"))
            matrix[score_index, context_index] = _to_float(best_row, "loocv_mae")

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(context_labels)))
    ax.set_xticklabels(context_labels)
    ax.set_yticks(np.arange(len(score_fields)))
    ax.set_yticklabels([SCORE_FIELD_LABELS[name] for name in score_fields])
    ax.set_title("Best MAE per Zoom after Choosing the Best Aggregation")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                "{:.1f}".format(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > np.mean(matrix) else "black",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("LOOCV MAE (m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

#123
def _plot_exact_distance_zoom_trends(
    output_path: Path,
    loo_rows: Sequence[Dict[str, Any]],
    best_by_context: Dict[str, Dict[str, Any]],
    ordered_contexts: Sequence[str],
) -> None:
    method_to_context = {str(row["method_id"]): context for context, row in best_by_context.items()}
    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in loo_rows:
        method_id = str(row["method_id"])
        if method_id not in method_to_context:
            continue
        context = method_to_context[method_id]
        distance = float(row["true_distance_m"])
        grouped[(context, distance)].append(float(row["absolute_error_m"]))

    distances = sorted({distance for _, distance in grouped.keys()})
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    for context_index, context in enumerate(ordered_contexts):
        y_values = []
        for distance in distances:
            errors = grouped[(context, distance)]
            y_values.append(sum(errors) / len(errors))
        ax.plot(
            distances,
            y_values,
            marker="o",
            linewidth=2,
            label=context,
            color=cmap(context_index % 10),
        )
    ax.set_title("Best Zoom-Specific Method MAE at Each Exact Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Mean absolute error (m)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_single_vs_fusion_exact_distance(
    output_path: Path,
    single_loo_rows: Sequence[Dict[str, Any]],
    fusion_loo_rows: Sequence[Dict[str, Any]],
    single_method_id: str,
    fusion_model_name: str,
) -> None:
    def _group(rows: Sequence[Dict[str, Any]], name_key: str, target_name: str) -> Tuple[List[float], List[float]]:
        grouped: Dict[float, List[float]] = defaultdict(list)
        for row in rows:
            if str(row[name_key]) != target_name:
                continue
            grouped[float(row["true_distance_m"])].append(float(row["absolute_error_m"]))
        distances = sorted(grouped.keys())
        maes = [sum(grouped[d]) / len(grouped[d]) for d in distances]
        return distances, maes

    single_distances, single_maes = _group(single_loo_rows, "method_id", single_method_id)
    fusion_distances, fusion_maes = _group(fusion_loo_rows, "model_name", fusion_model_name)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(single_distances, single_maes, marker="o", linewidth=2.2, label="Best single", color="#1f77b4")
    ax.plot(fusion_distances, fusion_maes, marker="o", linewidth=2.2, label="Best fusion", color="#d62728")
    ax.set_title("Best Single vs Best Fusion at Each Exact Distance")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Mean absolute error (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_context_winner_per_exact_distance(
    output_path: Path,
    loo_rows: Sequence[Dict[str, Any]],
    best_by_context: Dict[str, Dict[str, Any]],
) -> None:
    method_to_context = {str(row["method_id"]): context for context, row in best_by_context.items()}
    method_to_label = {
        str(row["method_id"]): "{} | {} | {}".format(
            row["context_window"],
            row["aggregation_method"],
            row["score_field"],
        )
        for row in best_by_context.values()
    }

    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in loo_rows:
        method_id = str(row["method_id"])
        if method_id not in method_to_context:
            continue
        context = method_to_context[method_id]
        distance = float(row["true_distance_m"])
        grouped[(context, distance)].append(float(row["absolute_error_m"]))

    distances = sorted({distance for _, distance in grouped.keys()})
    winners: List[Tuple[float, str, str, float]] = []
    for distance in distances:
        candidates: List[Tuple[float, str, str]] = []
        for method_id, context in method_to_context.items():
            errors = grouped[(context, distance)]
            mae = sum(errors) / len(errors)
            candidates.append((mae, context, method_id))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best_mae, best_context, best_method_id = candidates[0]
        winners.append((distance, best_context, best_method_id, best_mae))

    unique_contexts = sorted({context for _, context, _, _ in winners})
    cmap = plt.get_cmap("tab10")
    context_colors = {context: cmap(index % 10) for index, context in enumerate(unique_contexts)}

    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = np.arange(len(winners), dtype=np.float64)
    bar_colors = [context_colors[context] for _, context, _, _ in winners]
    maes = [mae for _, _, _, mae in winners]
    bars = ax.bar(x_positions, maes, color=bar_colors)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(distance)) for distance, _, _, _ in winners])
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Best zoom-specific MAE (m)")
    ax.set_title("Winning Zoom/Method at Each Exact Distance")
    ax.grid(axis="y", alpha=0.25)

    for bar, (_, context, method_id, mae) in zip(bars, winners):
        label = "{}\n{:.1f}m".format(context, mae)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.25,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=context_colors[context], label=context)
        for context in unique_contexts
    ]
    ax.legend(handles=legend_handles, title="Winning context", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extract_group_rows(
    rows: Sequence[Dict[str, Any]],
    model_name: str,
    grouping: str,
) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["model_name"]) == model_name and str(row["grouping"]) == grouping
    ]


def _plot_model_comparison_by_group(
    output_path: Path,
    grouped_rows: Sequence[Dict[str, Any]],
    model_names: Sequence[str],
    group_order: Sequence[str],
    grouping: str,
    value_key: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(group_order), dtype=np.float64)
    width = 0.38
    colors = ["#1f77b4", "#d62728"]
    for model_index, model_name in enumerate(model_names):
        model_rows = _extract_group_rows(grouped_rows, model_name, grouping)
        row_lookup = {}
        for row in model_rows:
            if grouping == "weather":
                group_key = str(row["weather"])
            elif grouping == "time_of_day":
                group_key = str(row["time_of_day"])
            else:
                group_key = str(row["distance_range"])
            row_lookup[group_key] = row
        values = [_to_float(row_lookup[group_key], value_key) for group_key in group_order]
        ax.bar(
            x + (model_index - 0.5) * width,
            values,
            width=width,
            label=model_name,
            color=colors[model_index % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_two_dimensional_matrix(
    rows: Sequence[Dict[str, Any]],
    model_name: str,
    grouping: str,
    row_order: Sequence[str],
    col_order: Sequence[str],
    value_key: str,
) -> np.ndarray:
    model_rows = _extract_group_rows(rows, model_name, grouping)
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in model_rows:
        if grouping == "weather__distance_range":
            lookup[(str(row["weather"]), str(row["distance_range"]))] = row
        elif grouping == "time_of_day__distance_range":
            lookup[(str(row["time_of_day"]), str(row["distance_range"]))] = row
    matrix = np.zeros((len(row_order), len(col_order)), dtype=np.float64)
    for row_index, row_name in enumerate(row_order):
        for col_index, col_name in enumerate(col_order):
            matrix[row_index, col_index] = _to_float(lookup[(row_name, col_name)], value_key)
    return matrix


def _plot_two_heatmaps(
    output_path: Path,
    grouped_rows_single: Sequence[Dict[str, Any]],
    grouped_rows_fusion: Sequence[Dict[str, Any]],
    single_model_name: str,
    fusion_model_name: str,
    grouping: str,
    row_order: Sequence[str],
    col_order: Sequence[str],
    value_key: str,
    row_label: str,
    col_label: str,
    title_prefix: str,
) -> None:
    single_matrix = _build_two_dimensional_matrix(
        grouped_rows_single,
        single_model_name,
        grouping,
        row_order,
        col_order,
        value_key,
    )
    fusion_matrix = _build_two_dimensional_matrix(
        grouped_rows_fusion,
        fusion_model_name,
        grouping,
        row_order,
        col_order,
        value_key,
    )
    difference_matrix = fusion_matrix - single_matrix

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    matrices = [
        (single_matrix, "{}: best single".format(title_prefix), "viridis_r"),
        (fusion_matrix, "{}: best fusion".format(title_prefix), "viridis_r"),
        (difference_matrix, "{}: fusion - single".format(title_prefix), "coolwarm"),
    ]
    for axis, (matrix, title, cmap) in zip(axes, matrices):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_xticks(np.arange(len(col_order)))
        axis.set_xticklabels(col_order)
        axis.set_yticks(np.arange(len(row_order)))
        axis.set_yticklabels(row_order)
        axis.set_title(title)
        axis.set_xlabel(col_label)
        axis.set_ylabel(row_label)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                axis.text(
                    col_index,
                    row_index,
                    "{:.1f}".format(matrix[row_index, col_index]),
                    ha="center",
                    va="center",
                    color="white" if (cmap == "viridis_r" and matrix[row_index, col_index] > np.mean(matrix)) else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_fusion_vs_top_single(
    output_path: Path,
    single_summary: Dict[str, Any],
    fusion_summary: Dict[str, Any],
) -> None:
    single = single_summary["winner"]
    fusion = fusion_summary["best_fusion_model"]
    labels = ["MAE", "RMSE", "Mean relative error", "Within 20m rate", "R²"]
    single_values = [
        float(single["loocv_mae"]),
        float(single["loocv_rmse"]),
        float(single["loocv_mean_relative_error"]),
        float(single["loocv_within_20m_rate"]),
        float(single["loocv_r2"]),
    ]
    fusion_values = [
        float(fusion["loocv_mae"]),
        float(fusion["loocv_rmse"]),
        float(fusion["loocv_mean_relative_error"]),
        float(fusion["loocv_within_20m_rate"]),
        float(fusion["loocv_r2"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    axes[0].bar(x[:3] - width / 2.0, single_values[:3], width=width, label="Best single", color="#1f77b4")
    axes[0].bar(x[:3] + width / 2.0, fusion_values[:3], width=width, label="Best fusion", color="#d62728")
    axes[0].set_xticks(x[:3])
    axes[0].set_xticklabels(labels[:3])
    axes[0].set_title("Error metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar([0 - width / 2.0, 1 - width / 2.0], [single_values[3], single_values[4]], width=width, color="#1f77b4")
    axes[1].bar([0 + width / 2.0, 1 + width / 2.0], [fusion_values[3], fusion_values[4]], width=width, color="#d62728")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels[3:])
    axes[1].set_title("Coverage and fit")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Best Single-Scale Method vs Best Multiscale Fusion Model", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _describe_zoom_findings(
    ordered_contexts: Sequence[str],
    single_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    best_by_context: Dict[str, Dict[str, Any]] = {}
    for row in single_rows:
        context = str(row["context_window"])
        if context not in best_by_context:
            best_by_context[context] = row
    lines.append("Best single method by zoom context:")
    for context in ordered_contexts:
        row = best_by_context[context]
        lines.append(
            "- `{}`: `{}` with MAE `{:.2f} m`".format(
                context,
                row["method_id"],
                float(row["loocv_mae"]),
            )
        )
    return lines


def _build_markdown_report(
    config: Dict[str, Any],
    single_summary: Dict[str, Any],
    fusion_summary: Dict[str, Any],
    method_rows: Sequence[Dict[str, Any]],
    grouped_rows: Sequence[Dict[str, Any]],
    fusion_grouped_rows: Sequence[Dict[str, Any]],
) -> str:
    ordered_contexts, _ = _context_order_and_labels(config)
    winner = single_summary["winner"]
    runner_up = single_summary["runner_up"]
    best_fusion = fusion_summary["best_fusion_model"]
    gap = fusion_summary["gap_vs_best_single_feature"]

    winner_name = str(winner["method_id"])
    fusion_name = str(best_fusion["model_name"])

    def _group_lookup(rows: Sequence[Dict[str, Any]], model_name: str, grouping: str, key_name: str) -> Dict[str, Dict[str, Any]]:
        result = {}
        for row in rows:
            if str(row["model_name"]) != model_name or str(row["grouping"]) != grouping:
                continue
            result[str(row[key_name])] = row
        return result

    winner_range = _group_lookup(grouped_rows, winner_name, "distance_range", "distance_range")
    fusion_range = _group_lookup(fusion_grouped_rows, fusion_name, "distance_range", "distance_range")
    winner_weather = _group_lookup(grouped_rows, winner_name, "weather", "weather")
    fusion_weather = _group_lookup(fusion_grouped_rows, fusion_name, "weather", "weather")
    winner_time = _group_lookup(grouped_rows, winner_name, "time_of_day", "time_of_day")
    fusion_time = _group_lookup(fusion_grouped_rows, fusion_name, "time_of_day", "time_of_day")

    lines: List[str] = []
    lines.append("# Study 03 Full Analysis")
    lines.append("")
    lines.append("## 1. Main Result")
    lines.append("")
    lines.append(
        "- Best single-scale method: `{}` with `MAE {:.2f} m`, `RMSE {:.2f} m`, `R^2 {:.3f}`".format(
            winner_name,
            float(winner["loocv_mae"]),
            float(winner["loocv_rmse"]),
            float(winner["loocv_r2"]),
        )
    )
    lines.append(
        "- Runner-up single-scale method: `{}` with `MAE {:.2f} m`".format(
            runner_up["method_id"],
            float(runner_up["loocv_mae"]),
        )
    )
    lines.append(
        "- Best multiscale fusion model: `{}` with `MAE {:.2f} m`, `RMSE {:.2f} m`, `R^2 {:.3f}`".format(
            fusion_name,
            float(best_fusion["loocv_mae"]),
            float(best_fusion["loocv_rmse"]),
            float(best_fusion["loocv_r2"]),
        )
    )
    lines.append(
        "- Fusion improvement vs best single: `MAE {:.2f} m`, `RMSE {:.2f} m`, `mean relative error {:.3f}`".format(
            float(gap["loocv_mae"]),
            float(gap["loocv_rmse"]),
            float(gap["loocv_mean_relative_error"]),
        )
    )
    lines.append("")
    lines.append("## 2. Zoom-Level Findings")
    lines.append("")
    lines.extend(_describe_zoom_findings(ordered_contexts, method_rows))
    lines.append("")
    lines.append(
        "The strongest single zoom remained `bbox_only`, while the strongest context-relative zoom remained `bbox_expand_1_5x`. "
        "The newly added wide contexts (`6x`, `8x`, `10x`, `16x`) all performed much worse than the tight local zooms."
    )
    lines.append("")
    lines.append("## 3. What Each Score Field Did as Zoom Changed")
    lines.append("")
    lines.append("- `object_depth` stayed strongest in the tightest crop, especially with `bbox_midpoint`.")
    lines.append("- `object_minus_ring` was strongest in medium local zooms such as `1.5x` and `2x`, where there is enough background for a meaningful ring.")
    lines.append("- `object_depth_percentile_5_95` helped some tight-crop variants, but did not beat the best raw-depth method.")
    lines.append("- `object_minus_ring_normalized` was usually weaker than the raw ring-difference version.")
    lines.append("- Very wide zooms caused all score fields to drift toward worse MAE because the drone signal became diluted by scene content.")
    lines.append("")
    lines.append("## 4. Distance-Range Behavior")
    lines.append("")
    for range_name in ("near", "mid", "far"):
        lines.append(
            "- `{}`: best single `MAE {:.2f} m` vs best fusion `MAE {:.2f} m`".format(
                range_name,
                float(winner_range[range_name]["mae"]),
                float(fusion_range[range_name]["mae"]),
            )
        )
    lines.append("")
    lines.append(
        "Interpretation: the fusion model mainly helped the `near` and `far` regimes, while `mid` became slightly worse. "
        "That means multiscale fusion reduced some of the distance-compression problem, but it did not solve it completely."
    )
    lines.append("")
    lines.append("## 5. Weather and Time-of-Day")
    lines.append("")
    for weather in ("clear_sky", "light_rain"):
        lines.append(
            "- `{}`: best single `MAE {:.2f} m` vs best fusion `MAE {:.2f} m`".format(
                weather,
                float(winner_weather[weather]["mae"]),
                float(fusion_weather[weather]["mae"]),
            )
        )
    for time_of_day in ("10AM", "8PM"):
        lines.append(
            "- `{}`: best single `MAE {:.2f} m` vs best fusion `MAE {:.2f} m`".format(
                time_of_day,
                float(winner_time[time_of_day]["mae"]),
                float(fusion_time[time_of_day]["mae"]),
            )
        )
    lines.append("")
    lines.append(
        "Interpretation: weather and time-of-day effects were smaller than the distance-range effects. "
        "The models changed only modestly across `clear_sky` vs `light_rain` and `10AM` vs `8PM`, while `near` and `far` remained the dominant challenge."
    )
    lines.append("")
    lines.append("## 6. What the Winning Fusion Model Actually Used")
    lines.append("")
    lines.append("- `bbox_only__bbox_midpoint__object_depth`")
    lines.append("- `bbox_only__inner50_median__object_depth_percentile_5_95`")
    lines.append("- `bbox_only__inner50_median__object_depth`")
    lines.append("- `bbox_only__bbox_midpoint__object_depth_percentile_5_95`")
    lines.append("- `bbox_expand_2x__inner50_median__object_minus_ring`")
    lines.append("- `bbox_expand_2x__bbox_midpoint__object_minus_ring`")
    lines.append("")
    lines.append(
        "That is the key lesson from Study 3: the model improved by combining a strong `bbox_only` core with `2x` local context correction. "
        "It did **not** improve by leaning on `6x`, `8x`, `10x`, `16x`, or `full_image`."
    )
    lines.append("")
    lines.append("## 7. Overall Conclusion")
    lines.append("")
    lines.append(
        "Study 3 provides evidence that a multiscale method can beat the best single zoom, but the useful multiscale combination is **local multiscale**, not broad scene-scale fusion. "
        "The strongest practical direction now is to build the next model around `bbox_only` depth features plus `2x` local contrast features."
    )
    lines.append("")
    lines.append("## 8. Graph Files")
    lines.append("")
    lines.append("- `best_mae_by_context.png`")
    lines.append("- `score_field_zoom_trends.png`")
    lines.append("- `context_score_heatmap.png`")
    lines.append("- `exact_distance_zoom_trends.png`")
    lines.append("- `best_context_winner_per_exact_distance.png`")
    lines.append("- `single_vs_fusion_summary.png`")
    lines.append("- `best_single_vs_fusion_exact_distance.png`")
    lines.append("- `distance_range_comparison.png`")
    lines.append("- `weather_comparison.png`")
    lines.append("- `time_of_day_comparison.png`")
    lines.append("- `weather_distance_mae_heatmaps.png`")
    lines.append("- `time_distance_mae_heatmaps.png`")
    return "\n".join(lines)


def run_analysis(config_path: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    config = load_config(config_path)
    output_root = Path(config["output_root"])
    reports_dir = output_root / "reports"
    analysis_dir = reports_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    single_summary = _read_json(reports_dir / "study_summary.json")
    fusion_summary = _read_json(reports_dir / "fusion_study_summary.json")
    method_rows = _sort_single_rows(_read_csv_rows(reports_dir / "method_metrics.csv"))
    grouped_rows = _read_csv_rows(reports_dir / "grouped_error_summary.csv")
    loo_rows = _read_csv_rows(reports_dir / "loo_predictions.csv")
    fusion_grouped_rows = _read_csv_rows(reports_dir / "fusion_grouped_error_summary.csv")
    fusion_loo_rows = _read_csv_rows(reports_dir / "fusion_loo_predictions.csv")

    ordered_contexts, context_labels = _context_order_and_labels(config)

    if verbose:
        print(
            "[study-03-analysis] Generating charts\n"
            "  reports_dir: {}\n"
            "  analysis_dir: {}\n"
            "  best_single: {}\n"
            "  best_fusion: {}".format(
                reports_dir,
                analysis_dir,
                single_summary["winner"]["method_id"],
                fusion_summary["best_fusion_model"]["model_name"],
            )
        )

    _plot_best_by_context(
        analysis_dir / "best_mae_by_context.png",
        ordered_contexts,
        context_labels,
        method_rows,
    )
    _plot_score_field_zoom_trends(
        analysis_dir / "score_field_zoom_trends.png",
        ordered_contexts,
        context_labels,
        method_rows,
    )
    _plot_context_score_heatmap(
        analysis_dir / "context_score_heatmap.png",
        ordered_contexts,
        context_labels,
        method_rows,
    )
    _plot_exact_distance_zoom_trends(
        analysis_dir / "exact_distance_zoom_trends.png",
        loo_rows,
        single_summary["best_by_context"],
        ordered_contexts,
    )
    _plot_best_context_winner_per_exact_distance(
        analysis_dir / "best_context_winner_per_exact_distance.png",
        loo_rows,
        single_summary["best_by_context"],
    )
    _plot_top_fusion_vs_top_single(
        analysis_dir / "single_vs_fusion_summary.png",
        single_summary,
        fusion_summary,
    )
    _plot_best_single_vs_fusion_exact_distance(
        analysis_dir / "best_single_vs_fusion_exact_distance.png",
        loo_rows,
        fusion_loo_rows,
        str(single_summary["winner"]["method_id"]),
        str(fusion_summary["best_fusion_model"]["model_name"]),
    )

    winner_name = str(single_summary["winner"]["method_id"])
    fusion_name = str(fusion_summary["best_fusion_model"]["model_name"])
    _plot_model_comparison_by_group(
        analysis_dir / "distance_range_comparison.png",
        grouped_rows + fusion_grouped_rows,
        [winner_name, fusion_name],
        ["near", "mid", "far"],
        "distance_range",
        "mae",
        "Distance-range MAE: best single vs best fusion",
        "MAE (m)",
    )
    _plot_model_comparison_by_group(
        analysis_dir / "weather_comparison.png",
        grouped_rows + fusion_grouped_rows,
        [winner_name, fusion_name],
        ["clear_sky", "light_rain"],
        "weather",
        "mae",
        "Weather MAE: best single vs best fusion",
        "MAE (m)",
    )
    _plot_model_comparison_by_group(
        analysis_dir / "time_of_day_comparison.png",
        grouped_rows + fusion_grouped_rows,
        [winner_name, fusion_name],
        ["10AM", "8PM"],
        "time_of_day",
        "mae",
        "Time-of-day MAE: best single vs best fusion",
        "MAE (m)",
    )
    _plot_two_heatmaps(
        analysis_dir / "weather_distance_mae_heatmaps.png",
        grouped_rows,
        fusion_grouped_rows,
        winner_name,
        fusion_name,
        "weather__distance_range",
        ["clear_sky", "light_rain"],
        ["near", "mid", "far"],
        "mae",
        "Weather",
        "Distance range",
        "Weather x distance MAE",
    )
    _plot_two_heatmaps(
        analysis_dir / "time_distance_mae_heatmaps.png",
        grouped_rows,
        fusion_grouped_rows,
        winner_name,
        fusion_name,
        "time_of_day__distance_range",
        ["10AM", "8PM"],
        ["near", "mid", "far"],
        "mae",
        "Time of day",
        "Distance range",
        "Time x distance MAE",
    )

    markdown_report = _build_markdown_report(
        config,
        single_summary,
        fusion_summary,
        method_rows,
        grouped_rows,
        fusion_grouped_rows,
    )
    report_path = analysis_dir / "study_03_analysis.md"
    _write_text(report_path, markdown_report)

    result = {
        "analysis_dir": str(analysis_dir),
        "markdown_report": str(report_path),
        "best_single_method_id": winner_name,
        "best_fusion_model_name": fusion_name,
        "chart_files": [
            "best_mae_by_context.png",
            "score_field_zoom_trends.png",
            "context_score_heatmap.png",
            "exact_distance_zoom_trends.png",
            "best_context_winner_per_exact_distance.png",
            "single_vs_fusion_summary.png",
            "best_single_vs_fusion_exact_distance.png",
            "distance_range_comparison.png",
            "weather_comparison.png",
            "time_of_day_comparison.png",
            "weather_distance_mae_heatmaps.png",
            "time_distance_mae_heatmaps.png",
        ],
    }
    if verbose:
        print(
            "[study-03-analysis] Analysis complete\n"
            "  markdown_report: {}\n"
            "  chart_count: {}".format(
                report_path,
                len(result["chart_files"]),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output and print only the final JSON result.",
    )
    args = parser.parse_args()
    result = run_analysis(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
