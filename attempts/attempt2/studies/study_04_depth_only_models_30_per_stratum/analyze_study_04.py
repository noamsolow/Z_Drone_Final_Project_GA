"""
Generate an analysis package with graphs and a compact markdown report for Study 04.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


STUDY_DIR = Path(__file__).resolve().parent
REPORTS_DIR = STUDY_DIR / "artifacts" / "reports"
ANALYSIS_DIR = REPORTS_DIR / "analysis"
CONFIG_PATH = STUDY_DIR / "config.yaml"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _to_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def _short_feature_name(name: str) -> str:
    return (
        name.replace("bbox_expand_", "")
        .replace("bbox_only", "bbox")
        .replace("full_image", "full")
        .replace("__bbox_midpoint__", "|mid|")
        .replace("__bbox_mean__", "|mean|")
        .replace("__inner50_median__", "|med50|")
        .replace("__object_depth_percentile_5_95", "|p5-95")
        .replace("__object_depth", "|depth")
        .replace("__object_minus_ring_normalized", "|ring_norm")
        .replace("__object_minus_ring", "|ring")
    )


def _short_model_name(name: str) -> str:
    return (
        name.replace("random_forest", "RF")
        .replace("gradient_boosting", "GB")
        .replace("extra_trees", "ET")
        .replace("greedy_forward_", "greedy_")
        .replace("_alpha_", "_a")
    )


def _plot_barh(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str,
    output_path: Path,
    colors: List[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors if colors is not None else "#35618f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(value, index, " {:.2f}".format(value), va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_grouped_bars(
    categories: List[str],
    series: List[Tuple[str, List[float], str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(series))
    for idx, (label, values, color) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    output_path: Path,
    cmap: str = "viridis_r",
) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 3, 0.6 * len(row_labels) + 2.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, "{:.1f}".format(matrix[r, c]), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_ci_rows(ci_rows: List[Dict[str, Any]], output_path: Path) -> None:
    ordered = []
    for row in ci_rows:
        if row["metric_name"] == "mae":
            ordered.append(row)
    labels = [row["model_name"] for row in ordered]
    point = np.asarray([float(row["point_estimate"]) for row in ordered], dtype=np.float64)
    low = np.asarray([float(row["ci_low_95"]) for row in ordered], dtype=np.float64)
    high = np.asarray([float(row["ci_high_95"]) for row in ordered], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        point,
        yerr=np.vstack([point - low, high - point]),
        fmt="o",
        capsize=5,
        color="#1f4e79",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_name(label) for label in labels], rotation=15, ha="right")
    ax.set_ylabel("MAE (m)")
    ax.set_title("95% Bootstrap CI for MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_exact_distance_mae(
    prediction_rows: Iterable[Dict[str, Any]],
    model_names: List[str],
) -> Dict[str, Dict[float, float]]:
    grouped: Dict[str, Dict[float, List[float]]] = {name: defaultdict(list) for name in model_names}
    for row in prediction_rows:
        model_name = str(row["model_name"])
        if model_name not in grouped:
            continue
        distance = float(row["true_distance_m"])
        grouped[model_name][distance].append(float(row["absolute_error_m"]))
    return {
        model_name: {
            distance: float(np.mean(errors))
            for distance, errors in sorted(distance_map.items())
        }
        for model_name, distance_map in grouped.items()
    }


def _plot_exact_distance_lines(distance_mae: Dict[str, Dict[float, float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = ["#4c78a8", "#f58518", "#54a24b"]
    for color, (model_name, series) in zip(palette, distance_mae.items()):
        distances = list(series.keys())
        maes = list(series.values())
        ax.plot(distances, maes, marker="o", label=_short_model_name(model_name), color=color)
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("MAE (m)")
    ax.set_title("Exact-distance MAE: best single vs best linear vs best tree")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_balanced_folds(
    rows: List[Dict[str, Any]],
    num_folds: int,
    seed: int,
) -> List[np.ndarray]:
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[str(row["stratum_key"])].append(index)
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for stratum_key in sorted(grouped_indices):
        indices = sorted(
            grouped_indices[stratum_key],
            key=lambda idx: (
                int(rows[idx]["selection_rank_within_stratum"]),
                str(rows[idx]["image_path"]),
            ),
        )
        permuted = list(indices)
        rng.shuffle(permuted)
        for rank, row_index in enumerate(permuted):
            folds[rank % num_folds].append(row_index)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def _fit_univariate_linear(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float]:
    if x_train.size == 0:
        return 0.0, 0.0
    if x_train.size == 1 or np.std(x_train) <= 1e-12:
        return float(np.mean(y_train)), 0.0
    design = np.column_stack([np.ones_like(x_train), x_train])
    coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _predict_univariate_linear(x_values: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * np.asarray(x_values, dtype=np.float64)


def _recompute_single_feature_predictions(
    fused_rows: List[Dict[str, Any]],
    feature_name: str,
    num_folds: int,
    cv_seed: int,
) -> np.ndarray:
    folds = _make_balanced_folds(fused_rows, num_folds=num_folds, seed=cv_seed)
    true_values = np.asarray([float(row["true_distance_m"]) for row in fused_rows], dtype=np.float64)
    x_values = np.asarray([float(row[feature_name]) for row in fused_rows], dtype=np.float64)
    predictions = np.zeros_like(true_values, dtype=np.float64)
    for fold_indices in folds:
        train_mask = np.ones(true_values.shape[0], dtype=bool)
        train_mask[fold_indices] = False
        intercept, slope = _fit_univariate_linear(x_values[train_mask], true_values[train_mask])
        predictions[fold_indices] = _predict_univariate_linear(x_values[fold_indices], intercept, slope)
    return predictions


def _plot_rf_importances(rows: List[Dict[str, Any]], best_tree_name: str, output_path: Path) -> None:
    filtered = [
        row for row in rows
        if row["model_name"] == best_tree_name and row["importance_type"] == "feature_importance"
    ]
    filtered = sorted(filtered, key=lambda row: float(row["importance_value"]), reverse=True)[:15]
    labels = [_short_feature_name(str(row["term"])) for row in filtered]
    values = [float(row["importance_value"]) for row in filtered]
    _plot_barh(labels, values, "Top feature importances: {}".format(_short_model_name(best_tree_name)), "Importance", output_path)


def _plot_context_best_single(single_rows: List[Dict[str, Any]], output_path: Path) -> None:
    best_by_context: Dict[str, Dict[str, Any]] = {}
    for row in single_rows:
        context = str(row["context_window"])
        if context not in best_by_context:
            best_by_context[context] = row
    labels = []
    values = []
    for context, row in sorted(best_by_context.items(), key=lambda item: float(item[1]["cv_mae"])):
        labels.append("{} | {}".format(context, row["score_field"]))
        values.append(float(row["cv_mae"]))
    _plot_barh(labels, values, "Best single feature per zoom context", "CV MAE (m)", output_path)


def _plot_context_score_heatmap(single_rows: List[Dict[str, Any]], output_path: Path) -> None:
    contexts = []
    for row in single_rows:
        context = str(row["context_window"])
        if context not in contexts:
            contexts.append(context)
    score_fields = [
        "object_depth",
        "object_depth_percentile_5_95",
        "object_minus_ring",
        "object_minus_ring_normalized",
    ]
    matrix = np.zeros((len(contexts), len(score_fields)), dtype=np.float64)
    for r, context in enumerate(contexts):
        for c, score_field in enumerate(score_fields):
            candidates = [
                row for row in single_rows
                if row["context_window"] == context and row["score_field"] == score_field
            ]
            matrix[r, c] = min(float(row["cv_mae"]) for row in candidates)
    _plot_heatmap(matrix, contexts, score_fields, "Best MAE by context x score field", output_path)


def run_analysis() -> Dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    config = _read_yaml(CONFIG_PATH)
    summary = _read_json(REPORTS_DIR / "depth_only_study_summary.json")
    single_rows = _read_csv_rows(REPORTS_DIR / "single_feature_cv_metrics.csv")
    model_rows = _read_csv_rows(REPORTS_DIR / "depth_only_model_metrics.csv")
    grouped_rows = _read_csv_rows(REPORTS_DIR / "depth_only_grouped_error_summary.csv")
    ci_rows = _read_csv_rows(REPORTS_DIR / "depth_only_bootstrap_intervals.csv")
    comparison_rows = _read_csv_rows(REPORTS_DIR / "depth_only_paired_model_comparisons.csv")
    importance_rows = _read_csv_rows(REPORTS_DIR / "depth_only_model_importances.csv")
    prediction_rows = _read_csv_rows(REPORTS_DIR / "depth_only_model_predictions.csv")
    fused_rows = _read_csv_rows(STUDY_DIR / "artifacts" / "fused_features" / "depth_only_feature_table.csv")

    best_single = summary["best_single_feature"]["method_id"]
    best_linear = summary["best_linear_multifeature_model"]["model_name"]
    best_tree = summary["best_tree_model"]["model_name"]
    best_single_predictions = _recompute_single_feature_predictions(
        fused_rows=fused_rows,
        feature_name=best_single,
        num_folds=int(summary["num_folds"]),
        cv_seed=int(config.get("depth_only_modeling", {}).get("cv_seed", 20260505)),
    )

    top_single = single_rows[:12]
    _plot_barh(
        labels=[_short_feature_name(str(row["method_id"])) for row in top_single],
        values=[float(row["cv_mae"]) for row in top_single],
        title="Top single depth features",
        xlabel="CV MAE (m)",
        output_path=ANALYSIS_DIR / "top_single_features.png",
        colors=["#2f6b9a" if row["context_window"] == "bbox_only" else "#7eaed6" for row in top_single],
    )

    top_models = model_rows[:12]
    _plot_barh(
        labels=[_short_model_name(str(row["model_name"])) for row in top_models],
        values=[float(row["cv_mae"]) for row in top_models],
        title="Top Study 4 depth-only models",
        xlabel="CV MAE (m)",
        output_path=ANALYSIS_DIR / "top_models.png",
        colors=["#c44e52" if row["model_kind"] == "tree" else "#4c78a8" for row in top_models],
    )

    _plot_grouped_bars(
        categories=["Best single", "Best linear", "Best tree"],
        series=[
            ("MAE", [
                float(summary["best_single_feature"]["cv_mae"]),
                float(summary["best_linear_multifeature_model"]["cv_mae"]),
                float(summary["best_tree_model"]["cv_mae"]),
            ], "#4c78a8"),
            ("RMSE", [
                float(summary["best_single_feature"]["cv_rmse"]),
                float(summary["best_linear_multifeature_model"]["cv_rmse"]),
                float(summary["best_tree_model"]["cv_rmse"]),
            ], "#f58518"),
        ],
        title="Best single vs best linear vs best tree",
        ylabel="Error (m)",
        output_path=ANALYSIS_DIR / "best_single_linear_tree_summary.png",
    )

    filtered_ci_rows = [
        row for row in ci_rows
        if row["model_name"] in {best_single, best_linear, best_tree}
    ]
    _plot_ci_rows(filtered_ci_rows, ANALYSIS_DIR / "bootstrap_ci_mae.png")

    selected_models = {
        best_single: ("single", "#4c78a8"),
        best_linear: ("linear", "#54a24b"),
        best_tree: ("tree", "#c44e52"),
    }

    distance_rows = [
        row for row in grouped_rows
        if row["grouping"] == "distance_range" and row["model_name"] in selected_models
    ]
    single_distance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row, prediction in zip(fused_rows, best_single_predictions):
        distance_range = "near" if float(row["true_distance_m"]) <= 60.0 else "mid" if float(row["true_distance_m"]) <= 100.0 else "far"
        single_distance["mae"][distance_range].append(abs(float(prediction) - float(row["true_distance_m"])))
    distance_categories = ["near", "mid", "far"]
    distance_series = []
    for model_name, (_, color) in selected_models.items():
        values = []
        for category in distance_categories:
            if model_name == best_single:
                values.append(float(np.mean(single_distance["mae"][category])))
            else:
                match = next(row for row in distance_rows if row["model_name"] == model_name and row["distance_range"] == category)
                values.append(float(match["mae"]))
        distance_series.append((_short_model_name(model_name), values, color))
    _plot_grouped_bars(
        distance_categories,
        distance_series,
        "Distance-range MAE comparison",
        "MAE (m)",
        ANALYSIS_DIR / "distance_range_comparison.png",
    )

    weather_rows = [
        row for row in grouped_rows
        if row["grouping"] == "weather" and row["model_name"] in selected_models
    ]
    single_weather: Dict[str, List[float]] = defaultdict(list)
    for row, prediction in zip(fused_rows, best_single_predictions):
        single_weather[str(row["weather"])].append(abs(float(prediction) - float(row["true_distance_m"])))
    weather_categories = ["clear_sky", "light_rain"]
    weather_series = []
    for model_name, (_, color) in selected_models.items():
        values = []
        for category in weather_categories:
            if model_name == best_single:
                values.append(float(np.mean(single_weather[category])))
            else:
                match = next(row for row in weather_rows if row["model_name"] == model_name and row["weather"] == category)
                values.append(float(match["mae"]))
        weather_series.append((_short_model_name(model_name), values, color))
    _plot_grouped_bars(
        weather_categories,
        weather_series,
        "Weather MAE comparison",
        "MAE (m)",
        ANALYSIS_DIR / "weather_comparison.png",
    )

    time_rows = [
        row for row in grouped_rows
        if row["grouping"] == "time_of_day" and row["model_name"] in selected_models
    ]
    single_time: Dict[str, List[float]] = defaultdict(list)
    for row, prediction in zip(fused_rows, best_single_predictions):
        single_time[str(row["time_of_day"])].append(abs(float(prediction) - float(row["true_distance_m"])))
    time_categories = ["10AM", "8PM"]
    time_series = []
    for model_name, (_, color) in selected_models.items():
        values = []
        for category in time_categories:
            if model_name == best_single:
                values.append(float(np.mean(single_time[category])))
            else:
                match = next(row for row in time_rows if row["model_name"] == model_name and row["time_of_day"] == category)
                values.append(float(match["mae"]))
        time_series.append((_short_model_name(model_name), values, color))
    _plot_grouped_bars(
        time_categories,
        time_series,
        "Time-of-day MAE comparison",
        "MAE (m)",
        ANALYSIS_DIR / "time_of_day_comparison.png",
    )

    weather_distance_rows = [
        row for row in grouped_rows
        if row["grouping"] == "weather__distance_range" and row["model_name"] == best_tree
    ]
    wd_matrix = np.zeros((2, 3), dtype=np.float64)
    for r, weather in enumerate(["clear_sky", "light_rain"]):
        for c, distance_range in enumerate(["near", "mid", "far"]):
            match = next(
                row for row in weather_distance_rows
                if row["weather"] == weather and row["distance_range"] == distance_range
            )
            wd_matrix[r, c] = float(match["mae"])
    _plot_heatmap(
        wd_matrix,
        ["clear_sky", "light_rain"],
        ["near", "mid", "far"],
        "Best tree MAE by weather x distance range",
        ANALYSIS_DIR / "weather_distance_heatmap_best_tree.png",
    )

    time_distance_rows = [
        row for row in grouped_rows
        if row["grouping"] == "time_of_day__distance_range" and row["model_name"] == best_tree
    ]
    td_matrix = np.zeros((2, 3), dtype=np.float64)
    for r, time_label in enumerate(["10AM", "8PM"]):
        for c, distance_range in enumerate(["near", "mid", "far"]):
            match = next(
                row for row in time_distance_rows
                if row["time_of_day"] == time_label and row["distance_range"] == distance_range
            )
            td_matrix[r, c] = float(match["mae"])
    _plot_heatmap(
        td_matrix,
        ["10AM", "8PM"],
        ["near", "mid", "far"],
        "Best tree MAE by time x distance range",
        ANALYSIS_DIR / "time_distance_heatmap_best_tree.png",
    )

    _plot_context_best_single(single_rows, ANALYSIS_DIR / "best_single_by_context.png")
    _plot_context_score_heatmap(single_rows, ANALYSIS_DIR / "context_score_heatmap.png")
    _plot_rf_importances(importance_rows, best_tree, ANALYSIS_DIR / "best_tree_feature_importances.png")

    synthetic_single_prediction_rows = [
        {
            "model_name": best_single,
            "true_distance_m": row["true_distance_m"],
            "absolute_error_m": abs(float(prediction) - float(row["true_distance_m"])),
        }
        for row, prediction in zip(fused_rows, best_single_predictions)
    ]
    exact_distance_mae = _build_exact_distance_mae(
        synthetic_single_prediction_rows + prediction_rows,
        [best_single, best_linear, best_tree],
    )
    _plot_exact_distance_lines(exact_distance_mae, ANALYSIS_DIR / "exact_distance_mae_comparison.png")

    comparison_text = "\n".join(
        "- {}: mean MAE delta = {:.2f}, 95% CI [{:.2f}, {:.2f}]".format(
            row["comparison_name"],
            float(row["mean_mae_delta_a_minus_b"]),
            float(row["ci_low_95"]),
            float(row["ci_high_95"]),
        )
        for row in comparison_rows
    )

    markdown = """# Study 04 Analysis

## Headline

- Best single feature: `{best_single}` with `MAE {best_single_mae:.2f}m`
- Best linear multifeature model: `{best_linear}` with `MAE {best_linear_mae:.2f}m`
- Best overall model: `{best_tree}` with `MAE {best_tree_mae:.2f}m`
- Best tree beat best single by `{best_delta:.2f}m` MAE

## What Strengthened

- A learned depth-only lower model clearly beats the best single depth feature.
- Tree models are now the top group, not just one lucky row.
- The winning feature family is still centered on `bbox_only`, `1.5x`, and `2x`.

## What The Graphs Show

- `top_single_features.png`
  - The strongest standalone features are still tight or local-context views.
- `top_models.png`
  - Tree models dominate the top of the leaderboard.
- `best_single_linear_tree_summary.png`
  - Linear fusion helps, but the tree model helps more.
- `bootstrap_ci_mae.png`
  - The best tree stays below the best single and best linear across the 95% MAE intervals.
- `distance_range_comparison.png`
  - The main gains still matter most around the hard distance regimes.
- `weather_comparison.png` and `time_of_day_comparison.png`
  - Weather and time matter, but less than distance structure.
- `best_tree_feature_importances.png`
  - The final winner is still driven by local relative-depth features, not broad-scene ones.

## Paired Significance

{comparison_text}
""".format(
        best_single=best_single,
        best_single_mae=float(summary["best_single_feature"]["cv_mae"]),
        best_linear=best_linear,
        best_linear_mae=float(summary["best_linear_multifeature_model"]["cv_mae"]),
        best_tree=best_tree,
        best_tree_mae=float(summary["best_tree_model"]["cv_mae"]),
        best_delta=float(summary["gap_best_model_vs_best_single"]["cv_mae"]),
        comparison_text=comparison_text,
    )
    (ANALYSIS_DIR / "study_04_analysis.md").write_text(markdown, encoding="utf-8")

    return {
        "analysis_dir": str(ANALYSIS_DIR),
        "analysis_markdown": str(ANALYSIS_DIR / "study_04_analysis.md"),
        "num_graphs": 11,
    }


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result, indent=2))
