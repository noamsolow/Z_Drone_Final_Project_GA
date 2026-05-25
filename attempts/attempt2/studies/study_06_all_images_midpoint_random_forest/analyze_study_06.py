"""
Generate a compact analysis package for Study 06.
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
import yaml


STUDY_DIR = Path(__file__).resolve().parent
REPORTS_DIR = STUDY_DIR / "artifacts" / "reports"
ANALYSIS_DIR = REPORTS_DIR / "analysis"
CONFIG_PATH = STUDY_DIR / "config.yaml"
STUDY_04_SUMMARY_PATH = (
    STUDY_DIR.parent
    / "study_04_depth_only_models_30_per_stratum"
    / "artifacts"
    / "reports"
    / "depth_only_study_summary.json"
)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        .replace("__object_depth", "|depth")
    )


def _short_context_name(name: str) -> str:
    return (
        name.replace("bbox_expand_", "")
        .replace("bbox_only", "bbox")
        .replace("full_image", "full")
    )


def _plot_barh(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    output_path: Path,
    colors: Sequence[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, values, color=list(colors) if colors is not None else "#2f6db3")
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
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
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
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_path: Path,
    cmap: str = "viridis_r",
) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 3, 0.75 * len(row_labels) + 2.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, "{:.1f}".format(matrix[r, c]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _group_prediction_rows(
    rows: Iterable[Dict[str, Any]],
    key_fn,
) -> Dict[Any, Dict[str, float]]:
    grouped_abs: Dict[Any, List[float]] = defaultdict(list)
    grouped_rel: Dict[Any, List[float]] = defaultdict(list)
    grouped_signed: Dict[Any, List[float]] = defaultdict(list)
    grouped_count: Dict[Any, int] = defaultdict(int)
    grouped_within10: Dict[Any, int] = defaultdict(int)
    grouped_within20: Dict[Any, int] = defaultdict(int)
    for row in rows:
        key = key_fn(row)
        abs_err = float(row["absolute_error_m"])
        rel_err = float(row["relative_error"])
        signed_err = float(row["signed_error_m"])
        grouped_abs[key].append(abs_err)
        grouped_rel[key].append(rel_err)
        grouped_signed[key].append(signed_err)
        grouped_count[key] += 1
        if abs_err <= 10.0:
            grouped_within10[key] += 1
        if abs_err <= 20.0:
            grouped_within20[key] += 1
    return {
        key: {
            "count": float(grouped_count[key]),
            "mae": float(np.mean(grouped_abs[key])),
            "mean_relative_error": float(np.mean(grouped_rel[key])),
            "mean_signed_error": float(np.mean(grouped_signed[key])),
            "within_10m_rate": float(grouped_within10[key] / grouped_count[key]),
            "within_20m_rate": float(grouped_within20[key] / grouped_count[key]),
        }
        for key in grouped_count
    }


def _plot_top_single_features(single_rows: List[Dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(single_rows, key=lambda row: int(float(row["rank"])))[:15]
    labels = [_short_feature_name(str(row["method_id"])) for row in ordered]
    values = [float(row["cv_mae"]) for row in ordered]
    _plot_barh(labels, values, "Top single features in Study 06", "CV MAE (m)", output_path)


def _plot_model_comparison(model_rows: List[Dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(model_rows, key=lambda row: int(float(row["rank"])))
    labels = [str(row["model_name"]) for row in ordered]
    maes = [float(row["cv_mae"]) for row in ordered]
    r2_values = [float(row["cv_r2"]) for row in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, maes, color=["#2f6db3", "#74a9cf"])
    axes[0].set_title("Random-forest MAE comparison")
    axes[0].set_ylabel("CV MAE (m)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, r2_values, color=["#c05a11", "#f4a259"])
    axes[1].set_title("Random-forest R^2 comparison")
    axes[1].set_ylabel("CV R^2")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_single_by_context(single_rows: List[Dict[str, Any]], output_path: Path) -> None:
    best_by_context: Dict[str, Dict[str, Any]] = {}
    for row in single_rows:
        context = str(row["context_window"])
        if context not in best_by_context:
            best_by_context[context] = row
    context_order = ["bbox_only", "bbox_expand_1_5x", "bbox_expand_2x", "bbox_expand_4x", "full_image"]
    labels = [_short_context_name(name) for name in context_order]
    values = [float(best_by_context[name]["cv_mae"]) for name in context_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#35618f")
    ax.set_title("Best single feature by context")
    ax.set_ylabel("CV MAE (m)")
    ax.grid(axis="y", alpha=0.25)
    for bar, name in zip(bars, context_order):
        row = best_by_context[name]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.2,
            str(row["aggregation_method"]).replace("inner50_median", "med50").replace("bbox_midpoint", "mid").replace("bbox_mean", "mean"),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_context_aggregation_heatmap(single_rows: List[Dict[str, Any]], output_path: Path) -> None:
    contexts = ["bbox_only", "bbox_expand_1_5x", "bbox_expand_2x", "bbox_expand_4x", "full_image"]
    aggregations = ["bbox_midpoint", "inner50_median", "bbox_mean"]
    matrix = np.zeros((len(contexts), len(aggregations)), dtype=np.float64)
    for r, context in enumerate(contexts):
        for c, aggregation in enumerate(aggregations):
            candidates = [
                row
                for row in single_rows
                if row["context_window"] == context and row["aggregation_method"] == aggregation
            ]
            matrix[r, c] = min(float(row["cv_mae"]) for row in candidates)
    _plot_heatmap(
        matrix,
        [_short_context_name(name) for name in contexts],
        ["mid", "med50", "mean"],
        "Single-feature MAE by context and aggregation",
        output_path,
    )


def _plot_rf_importances(importance_rows: List[Dict[str, Any]], output_path: Path) -> None:
    filtered = [
        row
        for row in importance_rows
        if row["model_name"] == "random_forest_top_24" and row["importance_type"] == "feature_importance"
    ]
    filtered = sorted(filtered, key=lambda row: float(row["importance_value"]), reverse=True)
    labels = [_short_feature_name(str(row["term"])) for row in filtered]
    values = [float(row["importance_value"]) for row in filtered]
    _plot_barh(labels, values, "Feature importances: Study 06 best random forest", "Importance", output_path)


def _plot_exact_distance_curves(prediction_rows: List[Dict[str, Any]], output_path: Path) -> Dict[float, Dict[str, float]]:
    grouped = _group_prediction_rows(prediction_rows, key_fn=lambda row: float(row["true_distance_m"]))
    ordered_distances = sorted(grouped)
    mae = [grouped[d]["mae"] for d in ordered_distances]
    signed = [grouped[d]["mean_signed_error"] for d in ordered_distances]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(ordered_distances, mae, marker="o", color="#2f6db3", linewidth=2)
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("Best random forest by exact distance")
    axes[0].grid(alpha=0.25)
    axes[1].plot(ordered_distances, signed, marker="o", color="#c05a11", linewidth=2)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_xlabel("True distance (m)")
    axes[1].set_ylabel("Mean signed error (m)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return grouped


def _plot_weather_time_comparison(prediction_rows: List[Dict[str, Any]], output_path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    weather = _group_prediction_rows(prediction_rows, key_fn=lambda row: str(row["weather"]))
    time_of_day = _group_prediction_rows(prediction_rows, key_fn=lambda row: str(row["time_of_day"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    weather_keys = sorted(weather)
    time_keys = sorted(time_of_day)
    axes[0].bar(weather_keys, [weather[key]["mae"] for key in weather_keys], color="#54a24b")
    axes[0].set_title("MAE by weather")
    axes[0].set_ylabel("MAE (m)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(time_keys, [time_of_day[key]["mae"] for key in time_keys], color="#dd8452")
    axes[1].set_title("MAE by time of day")
    axes[1].set_ylabel("MAE (m)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return weather, time_of_day


def _plot_weather_time_heatmap(prediction_rows: List[Dict[str, Any]], output_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped = _group_prediction_rows(
        prediction_rows,
        key_fn=lambda row: (str(row["weather"]), str(row["time_of_day"])),
    )
    weather_keys = sorted({key[0] for key in grouped})
    time_keys = sorted({key[1] for key in grouped})
    matrix = np.zeros((len(weather_keys), len(time_keys)), dtype=np.float64)
    for r, weather in enumerate(weather_keys):
        for c, time_key in enumerate(time_keys):
            matrix[r, c] = grouped[(weather, time_key)]["mae"]
    _plot_heatmap(matrix, weather_keys, time_keys, "Best random forest MAE by weather and time", output_path)
    return grouped


def _describe_distance_behavior(distance_grouped: Dict[float, Dict[str, float]]) -> List[str]:
    ordered = sorted(distance_grouped.items(), key=lambda item: item[0])
    best_distance, best_metrics = min(ordered, key=lambda item: item[1]["mae"])
    worst_distance, worst_metrics = max(ordered, key=lambda item: item[1]["mae"])
    lines = []
    lines.append(
        "- Best exact-distance region: `{:.0f}m` with `MAE {:.2f}m`.".format(
            best_distance,
            best_metrics["mae"],
        )
    )
    lines.append(
        "- Worst exact-distance region: `{:.0f}m` with `MAE {:.2f}m`.".format(
            worst_distance,
            worst_metrics["mae"],
        )
    )
    near = distance_grouped[min(distance_grouped)]
    far = distance_grouped[max(distance_grouped)]
    lines.append(
        "- Near distances show overprediction: at `{:.0f}m`, mean signed error is `{:.2f}m`.".format(
            min(distance_grouped),
            near["mean_signed_error"],
        )
    )
    lines.append(
        "- Far distances show underprediction: at `{:.0f}m`, mean signed error is `{:.2f}m`.".format(
            max(distance_grouped),
            far["mean_signed_error"],
        )
    )
    return lines


def run_analysis() -> Dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    config = _read_yaml(CONFIG_PATH)
    summary = _read_json(REPORTS_DIR / "summary.json")
    single_rows = _read_csv_rows(REPORTS_DIR / "single_feature_cv_metrics.csv")
    model_rows = _read_csv_rows(REPORTS_DIR / "subset_random_forest_metrics.csv")
    importance_rows = _read_csv_rows(REPORTS_DIR / "subset_random_forest_feature_importances.csv")
    prediction_rows = _read_csv_rows(REPORTS_DIR / "subset_random_forest_predictions.csv")

    study_04_summary = _read_json(STUDY_04_SUMMARY_PATH) if STUDY_04_SUMMARY_PATH.exists() else None

    best_model_name = str(summary["best_random_forest_model"])
    best_model_prediction_rows = [row for row in prediction_rows if row["model_name"] == best_model_name]

    _plot_top_single_features(single_rows, ANALYSIS_DIR / "top_single_features.png")
    _plot_model_comparison(model_rows, ANALYSIS_DIR / "top_random_forest_models.png")
    _plot_best_single_by_context(single_rows, ANALYSIS_DIR / "best_single_by_context.png")
    _plot_context_aggregation_heatmap(single_rows, ANALYSIS_DIR / "context_aggregation_heatmap.png")
    _plot_rf_importances(importance_rows, ANALYSIS_DIR / "best_random_forest_feature_importances.png")
    distance_grouped = _plot_exact_distance_curves(
        best_model_prediction_rows,
        ANALYSIS_DIR / "exact_distance_behavior.png",
    )
    weather_grouped, time_grouped = _plot_weather_time_comparison(
        best_model_prediction_rows,
        ANALYSIS_DIR / "weather_time_comparison.png",
    )
    weather_time_grouped = _plot_weather_time_heatmap(
        best_model_prediction_rows,
        ANALYSIS_DIR / "weather_time_heatmap.png",
    )

    top_single = single_rows[0]
    top_model = model_rows[0]
    report_lines: List[str] = []
    report_lines.append("# Study 06 Analysis")
    report_lines.append("")
    report_lines.append("## Headline")
    report_lines.append("")
    report_lines.append(
        "- Best single feature: `{}` with `CV MAE {:.2f}m`.".format(
            top_single["method_id"],
            _to_float(top_single, "cv_mae"),
        )
    )
    report_lines.append(
        "- Best random forest: `{}` with `CV MAE {:.2f}m`, `RMSE {:.2f}m`, `R^2 {:.3f}`.".format(
            top_model["model_name"],
            _to_float(top_model, "cv_mae"),
            _to_float(top_model, "cv_rmse"),
            _to_float(top_model, "cv_r2"),
        )
    )
    improvement = _to_float(top_model, "cv_mae") - _to_float(top_single, "cv_mae")
    report_lines.append(
        "- Improvement vs best single: `MAE {:.2f}m`.".format(improvement)
    )
    if study_04_summary is not None:
        study_04_best = float(study_04_summary["best_model"]["cv_mae"])
        report_lines.append(
            "- Relative to Study 04 best model, Study 06 changes MAE by `{:.2f}m`.".format(
                _to_float(top_model, "cv_mae") - study_04_best
            )
        )
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append("")
    report_lines.append(
        "- `num_image_rows = {}`".format(int(summary["num_image_rows"]))
    )
    report_lines.append(
        "- `num_available_features = {}`".format(int(summary["num_available_features"]))
    )
    report_lines.append(
        "- Score fields kept: `{}`".format("`, `".join(str(item) for item in summary["score_fields"]))
    )
    report_lines.append("- Context subset came from the Study 06 config.")
    report_lines.append("")
    report_lines.append("## Single-feature story")
    report_lines.append("")
    report_lines.append(
        "- The strongest single feature remained the same core winner from Study 04: `bbox_only__bbox_midpoint__object_depth`."
    )
    report_lines.append(
        "- The next-best alternatives stayed local: `1.5x`, `bbox_only med50`, and `2x midpoint`."
    )
    report_lines.append(
        "- `full_image` remained weak, and `bbox_mean` remained the weakest aggregation family."
    )
    report_lines.append("")
    report_lines.append("## Model story")
    report_lines.append("")
    report_lines.append(
        "- The random forest still beat the best single feature by a clear margin."
    )
    report_lines.append(
        "- The `top_24` name is a legacy label; in this study it effectively used all `15` available features."
    )
    report_lines.append(
        "- Adding the last few lower-ranked features still helped slightly: `top_24` beat `top_12` on both MAE and RMSE."
    )
    report_lines.append("")
    report_lines.append("## Distance behavior")
    report_lines.append("")
    report_lines.extend(_describe_distance_behavior(distance_grouped))
    report_lines.append(
        "- The middle band around `80m` to `100m` is much easier than the nearest and farthest distances."
    )
    report_lines.append(
        "- The overall gain is real, but the model still shows strong distance compression."
    )
    report_lines.append("")
    report_lines.append("## Condition behavior")
    report_lines.append("")
    best_weather = min(weather_grouped.items(), key=lambda item: item[1]["mae"])
    worst_weather = max(weather_grouped.items(), key=lambda item: item[1]["mae"])
    best_time = min(time_grouped.items(), key=lambda item: item[1]["mae"])
    worst_time = max(time_grouped.items(), key=lambda item: item[1]["mae"])
    worst_weather_time = max(weather_time_grouped.items(), key=lambda item: item[1]["mae"])
    report_lines.append(
        "- Best weather condition: `{}` with `MAE {:.2f}m`.".format(best_weather[0], best_weather[1]["mae"])
    )
    report_lines.append(
        "- Worst weather condition: `{}` with `MAE {:.2f}m`.".format(worst_weather[0], worst_weather[1]["mae"])
    )
    report_lines.append(
        "- Best time-of-day condition: `{}` with `MAE {:.2f}m`.".format(best_time[0], best_time[1]["mae"])
    )
    report_lines.append(
        "- Worst time-of-day condition: `{}` with `MAE {:.2f}m`.".format(worst_time[0], worst_time[1]["mae"])
    )
    report_lines.append(
        "- Hardest combined cell: `{} + {}` with `MAE {:.2f}m`.".format(
            worst_weather_time[0][0],
            worst_weather_time[0][1],
            worst_weather_time[1]["mae"],
        )
    )
    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("")
    report_lines.append(
        "- Study 06 supports the same local-depth story as Study 04, but under a much larger and simpler full-data setting."
    )
    report_lines.append(
        "- The model does not need percentile or ring features to remain useful; raw `object_depth` alone still carries a strong signal."
    )
    report_lines.append(
        "- The remaining weakness is not the lack of a depth signal, but the imperfect conversion from relative depth to metric distance at the extremes."
    )
    report_lines.append("")
    report_lines.append("## Files Generated")
    report_lines.append("")
    report_lines.append("- `top_single_features.png`")
    report_lines.append("- `top_random_forest_models.png`")
    report_lines.append("- `best_single_by_context.png`")
    report_lines.append("- `context_aggregation_heatmap.png`")
    report_lines.append("- `best_random_forest_feature_importances.png`")
    report_lines.append("- `exact_distance_behavior.png`")
    report_lines.append("- `weather_time_comparison.png`")
    report_lines.append("- `weather_time_heatmap.png`")
    report_lines.append("- `study_06_analysis.md`")

    report_path = ANALYSIS_DIR / "study_06_analysis.md"
    _write_text(report_path, "\n".join(report_lines))

    return {
        "analysis_dir": str(ANALYSIS_DIR),
        "report_path": str(report_path),
        "generated_files": [
            "top_single_features.png",
            "top_random_forest_models.png",
            "best_single_by_context.png",
            "context_aggregation_heatmap.png",
            "best_random_forest_feature_importances.png",
            "exact_distance_behavior.png",
            "weather_time_comparison.png",
            "weather_time_heatmap.png",
            "study_06_analysis.md",
        ],
        "summary": summary,
        "config": config,
    }


def main() -> None:
    result = run_analysis()
    print("Study 06 analysis package written to {}".format(result["analysis_dir"]))


if __name__ == "__main__":
    main()
