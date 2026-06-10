"""
Run per-drone calibration for Attempt 8 real Nenrus predictions.

The calibration core is intentionally reused from Attempt 5 Study04 so that the
comparison is methodological apples-to-apples.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
ATTEMPT5_CALIBRATION_SCRIPT = REPO_ROOT / "attempts" / "attempt5" / "studies" / "study04" / "run_nenrus_calibration.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Attempt 8 Nenrus per-drone calibration study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML did not contain an object: {}".format(path))
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_attempt5_calibration_module() -> Any:
    spec = importlib.util.spec_from_file_location("attempt5_study04_calibration", ATTEMPT5_CALIBRATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Attempt 5 calibration script: {}".format(ATTEMPT5_CALIBRATION_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_float(value: Any) -> float:
    return float(value)


def _find_summary_row(
    rows: Sequence[Dict[str, Any]],
    calibration_model: str,
    calibration_fraction: float,
) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("calibration_model")) != calibration_model:
            continue
        if abs(float(row.get("calibration_fraction")) - float(calibration_fraction)) < 1e-9:
            return dict(row)
    raise KeyError("Missing {} @ {}".format(calibration_model, calibration_fraction))


def _plot_grouped_bars(
    labels: Sequence[str],
    attempt5_values: Sequence[float],
    attempt8_values: Sequence[float],
    output_path: Path,
    title: str,
    ylabel: str = "MAE (m)",
) -> None:
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars_5 = ax.bar(x - width / 2, attempt5_values, width, label="Attempt 5", color="#8da0cb")
    bars_8 = ax.bar(x + width / 2, attempt8_values, width, label="Attempt 8", color="#fc8d62")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ymax = max([*attempt5_values, *attempt8_values, 1.0])
    ax.set_ylim(0.0, ymax * 1.18)
    for bars in (bars_5, bars_8):
        for bar in bars:
            height = float(bar.get_height())
            ax.annotate(
                "{:.2f}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_selected_fraction_track(
    attempt5_rows: Sequence[Dict[str, Any]],
    attempt8_rows: Sequence[Dict[str, Any]],
    selected_model: str,
    output_path: Path,
) -> None:
    attempt5_by_fraction = {
        float(row["calibration_fraction"]): float(row["mae_mean"])
        for row in attempt5_rows
        if str(row["calibration_model"]) == selected_model
    }
    attempt8_by_fraction = {
        float(row["calibration_fraction"]): float(row["mae_mean"])
        for row in attempt8_rows
        if str(row["calibration_model"]) == selected_model
    }
    fractions = sorted(set(attempt5_by_fraction) & set(attempt8_by_fraction))
    labels = ["{}%".format(int(round(frac * 100))) for frac in fractions]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(labels, [attempt5_by_fraction[frac] for frac in fractions], marker="o", linewidth=2.2, label="Attempt 5")
    ax.plot(labels, [attempt8_by_fraction[frac] for frac in fractions], marker="o", linewidth=2.2, label="Attempt 8")
    ax.set_title("Per-Drone Quadratic Calibration Across Calibration Set Sizes")
    ax.set_xlabel("Calibration fraction")
    ax.set_ylabel("Mean test MAE across 100 splits (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_20pct_model_comparison(
    attempt5_rows: Sequence[Dict[str, Any]],
    attempt8_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    pretty = {
        "raw_uncalibrated": "Raw",
        "global_affine": "Global affine",
        "global_quadratic": "Global quadratic",
        "global_scale_only": "Scale only",
        "per_drone_affine": "Per-drone affine",
        "per_drone_quadratic": "Per-drone quadratic",
    }
    attempt5_20 = {
        str(row["calibration_model"]): float(row["mae_mean"])
        for row in attempt5_rows
        if abs(float(row["calibration_fraction"]) - 0.20) < 1e-9
    }
    attempt8_20 = {
        str(row["calibration_model"]): float(row["mae_mean"])
        for row in attempt8_rows
        if abs(float(row["calibration_fraction"]) - 0.20) < 1e-9
    }
    models = [name for name in pretty if name in attempt5_20 and name in attempt8_20]
    _plot_grouped_bars(
        labels=[pretty[name] for name in models],
        attempt5_values=[attempt5_20[name] for name in models],
        attempt8_values=[attempt8_20[name] for name in models],
        output_path=output_path,
        title="Calibration Models at 20% Calibration Data",
    )


def _plot_mae_vs_rmse(
    split_rows: Sequence[Dict[str, Any]],
    raw_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    selected_model: str,
    output_path: Path,
) -> None:
    model_colors = {
        "raw_uncalibrated": "#6b7280",
        "global_affine": "#2563eb",
        "global_quadratic": "#0891b2",
        "global_scale_only": "#7c3aed",
        "per_drone_affine": "#ea580c",
        "per_drone_quadratic": "#059669",
    }
    fraction_markers = {
        0.10: "o",
        0.20: "s",
        0.30: "^",
        0.50: "D",
    }
    model_labels = {
        "raw_uncalibrated": "Raw",
        "global_affine": "Global affine",
        "global_quadratic": "Global quadratic",
        "global_scale_only": "Scale only",
        "per_drone_affine": "Per-drone affine",
        "per_drone_quadratic": "Per-drone quadratic",
    }

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.set_title("Attempt 8 Study02: MAE vs RMSE")
    ax.set_xlabel("MAE (m)")
    ax.set_ylabel("RMSE (m)")
    ax.grid(alpha=0.25)

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    mae_values = [float(raw_metrics["mae"]), float(final_metrics["mae"])]
    rmse_values = [float(raw_metrics["rmse"]), float(final_metrics["rmse"])]
    for row in split_rows:
        model_name = str(row["calibration_model"])
        by_model.setdefault(model_name, []).append(dict(row))
        mae_values.append(float(row["mae_mean"]))
        rmse_values.append(float(row["rmse_mean"]))

    max_axis = max([*mae_values, *rmse_values, 1.0]) * 1.08
    ax.plot(
        [0.0, max_axis],
        [0.0, max_axis],
        linestyle="--",
        linewidth=1.3,
        color="#cbd5e1",
        label="RMSE = MAE",
        zorder=1,
    )

    for model_name, model_rows in sorted(by_model.items()):
        sorted_rows = sorted(model_rows, key=lambda row: float(row["calibration_fraction"]))
        color = model_colors.get(model_name, "#334155")
        x_values = [float(row["mae_mean"]) for row in sorted_rows]
        y_values = [float(row["rmse_mean"]) for row in sorted_rows]
        ax.plot(x_values, y_values, color=color, linewidth=1.8, alpha=0.8, zorder=2)
        for row in sorted_rows:
            fraction = round(float(row["calibration_fraction"]), 2)
            x_value = float(row["mae_mean"])
            y_value = float(row["rmse_mean"])
            marker = fraction_markers.get(fraction, "o")
            size = 90 if model_name == selected_model else 62
            ax.scatter(
                x_value,
                y_value,
                s=size,
                marker=marker,
                color=color,
                edgecolors="#ffffff",
                linewidths=0.8,
                zorder=3,
            )
            if model_name == selected_model:
                ax.annotate(
                    "{}%".format(int(round(fraction * 100))),
                    (x_value, y_value),
                    xytext=(6, -12),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    weight="bold",
                )

    raw_x = float(raw_metrics["mae"])
    raw_y = float(raw_metrics["rmse"])
    final_x = float(final_metrics["mae"])
    final_y = float(final_metrics["rmse"])
    ax.scatter(
        raw_x,
        raw_y,
        s=160,
        marker="X",
        color="#111827",
        edgecolors="#ffffff",
        linewidths=0.9,
        zorder=4,
    )
    ax.annotate(
        "raw full set",
        (raw_x, raw_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color="#111827",
        weight="bold",
    )
    ax.scatter(
        final_x,
        final_y,
        s=180,
        marker="*",
        color="#dc2626",
        edgecolors="#ffffff",
        linewidths=0.9,
        zorder=4,
    )
    ax.annotate(
        "final full-data fit",
        (final_x, final_y),
        xytext=(10, -14),
        textcoords="offset points",
        fontsize=8,
        color="#991b1b",
        weight="bold",
    )

    ax.set_xlim(0.0, max_axis)
    ax.set_ylim(0.0, max_axis)

    model_handles = [
        Line2D([0], [0], color=model_colors.get(model_name, "#334155"), linewidth=2.0, label=model_labels.get(model_name, model_name))
        for model_name in sorted(by_model)
    ]
    fraction_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#374151",
            linestyle="None",
            markersize=8,
            markerfacecolor="#ffffff",
            label="{}% calibration".format(int(round(fraction * 100))),
        )
        for fraction, marker in sorted(fraction_markers.items())
    ]
    special_handles = [
        Line2D([0], [0], marker="X", color="#111827", linestyle="None", markersize=9, label="Raw full set"),
        Line2D([0], [0], marker="*", color="#dc2626", linestyle="None", markersize=11, label="Final full-data fit"),
    ]

    first_legend = ax.legend(handles=model_handles, loc="upper left", frameon=True, title="Calibration model")
    ax.add_artist(first_legend)
    ax.legend(handles=[*fraction_handles, *special_handles], loc="lower right", frameon=True, title="Markers")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_comparison(
    config: Dict[str, Any],
    attempt8_summary: Dict[str, Any],
    reports_dir: Path,
    plots_dir: Path,
) -> Dict[str, Any]:
    comparison_config = dict(config.get("comparison", {}))
    selected_model = str(config["calibration"].get("selected_model", "per_drone_quadratic"))
    reference_fraction = 0.20

    attempt5_summary = _read_json(_resolve_repo_path(str(comparison_config["attempt5_summary_json"])))
    attempt5_split = _read_csv(_resolve_repo_path(str(comparison_config["attempt5_split_eval_summary_csv"])))
    attempt5_by_drone = _read_csv(_resolve_repo_path(str(comparison_config["attempt5_before_after_by_drone_type_csv"])))

    attempt8_split = _read_csv(reports_dir / "split_eval_summary.csv")
    attempt8_by_drone = _read_csv(reports_dir / "before_after_by_drone_type.csv")

    attempt5_20 = _find_summary_row(attempt5_split, selected_model, reference_fraction)
    attempt8_20 = _find_summary_row(attempt8_split, selected_model, reference_fraction)

    overall_rows = [
        {
            "stage": "raw_uncalibrated_full_dataset",
            "attempt5_mae": attempt5_summary["raw_uncalibrated_metrics"]["mae"],
            "attempt8_mae": attempt8_summary["raw_uncalibrated_metrics"]["mae"],
        },
        {
            "stage": "honest_split_20pct_calibration_test_mean",
            "attempt5_mae": attempt5_20["mae_mean"],
            "attempt8_mae": attempt8_20["mae_mean"],
        },
        {
            "stage": "final_full_data_per_drone_quadratic_fit",
            "attempt5_mae": attempt5_summary["final_metrics_full_data_fit"]["mae"],
            "attempt8_mae": attempt8_summary["final_metrics_full_data_fit"]["mae"],
        },
    ]
    for row in overall_rows:
        row["attempt5_mae"] = float(row["attempt5_mae"])
        row["attempt8_mae"] = float(row["attempt8_mae"])
        row["attempt8_minus_attempt5_mae"] = float(row["attempt8_mae"] - row["attempt5_mae"])
        row["attempt8_percent_change_vs_attempt5"] = (
            float(row["attempt8_minus_attempt5_mae"] / row["attempt5_mae"] * 100.0)
            if float(row["attempt5_mae"]) != 0.0
            else None
        )

    attempt5_by_drone_map = {str(row["drone_type"]): row for row in attempt5_by_drone}
    by_drone_rows: List[Dict[str, Any]] = []
    for attempt8_row in attempt8_by_drone:
        drone_type = str(attempt8_row["drone_type"])
        attempt5_row = attempt5_by_drone_map.get(drone_type)
        if not attempt5_row:
            continue
        output_row = {
            "drone_type": drone_type,
            "count": int(float(attempt8_row["count"])),
            "attempt5_raw_mae": _as_float(attempt5_row["raw_mae"]),
            "attempt8_raw_mae": _as_float(attempt8_row["raw_mae"]),
            "attempt8_minus_attempt5_raw_mae": _as_float(attempt8_row["raw_mae"]) - _as_float(attempt5_row["raw_mae"]),
            "attempt5_calibrated_mae": _as_float(attempt5_row["calibrated_mae"]),
            "attempt8_calibrated_mae": _as_float(attempt8_row["calibrated_mae"]),
            "attempt8_minus_attempt5_calibrated_mae": _as_float(attempt8_row["calibrated_mae"])
            - _as_float(attempt5_row["calibrated_mae"]),
        }
        by_drone_rows.append(output_row)

    model_20_rows: List[Dict[str, Any]] = []
    attempt5_model_20 = {
        str(row["calibration_model"]): row
        for row in attempt5_split
        if abs(float(row["calibration_fraction"]) - reference_fraction) < 1e-9
    }
    for attempt8_row in attempt8_split:
        if abs(float(attempt8_row["calibration_fraction"]) - reference_fraction) >= 1e-9:
            continue
        model_name = str(attempt8_row["calibration_model"])
        attempt5_row = attempt5_model_20.get(model_name)
        if not attempt5_row:
            continue
        model_20_rows.append(
            {
                "calibration_model": model_name,
                "calibration_fraction": reference_fraction,
                "attempt5_mae_mean": _as_float(attempt5_row["mae_mean"]),
                "attempt8_mae_mean": _as_float(attempt8_row["mae_mean"]),
                "attempt8_minus_attempt5_mae_mean": _as_float(attempt8_row["mae_mean"]) - _as_float(attempt5_row["mae_mean"]),
                "attempt5_within_10m_rate_mean": _as_float(attempt5_row["within_10m_rate_mean"]),
                "attempt8_within_10m_rate_mean": _as_float(attempt8_row["within_10m_rate_mean"]),
            }
        )

    _write_csv(reports_dir / "comparison_vs_attempt5.csv", overall_rows)
    _write_csv(reports_dir / "comparison_by_drone_type_vs_attempt5.csv", by_drone_rows)
    _write_csv(reports_dir / "comparison_calibration_models_20pct_vs_attempt5.csv", model_20_rows)

    _plot_grouped_bars(
        labels=["Raw", "20% split", "Final fit"],
        attempt5_values=[float(row["attempt5_mae"]) for row in overall_rows],
        attempt8_values=[float(row["attempt8_mae"]) for row in overall_rows],
        output_path=plots_dir / "comparison_overall_mae_vs_attempt5.png",
        title="Attempt 8 vs Attempt 5 on Real Nenrus Calibration",
    )
    _plot_grouped_bars(
        labels=[str(row["drone_type"]) + " raw" for row in by_drone_rows]
        + [str(row["drone_type"]) + " calibrated" for row in by_drone_rows],
        attempt5_values=[float(row["attempt5_raw_mae"]) for row in by_drone_rows]
        + [float(row["attempt5_calibrated_mae"]) for row in by_drone_rows],
        attempt8_values=[float(row["attempt8_raw_mae"]) for row in by_drone_rows]
        + [float(row["attempt8_calibrated_mae"]) for row in by_drone_rows],
        output_path=plots_dir / "comparison_by_drone_type_vs_attempt5.png",
        title="Raw and Calibrated MAE by Drone Type",
    )
    _plot_selected_fraction_track(
        attempt5_rows=attempt5_split,
        attempt8_rows=attempt8_split,
        selected_model=selected_model,
        output_path=plots_dir / "comparison_selected_calibration_fraction_track_vs_attempt5.png",
    )
    _plot_20pct_model_comparison(
        attempt5_rows=attempt5_split,
        attempt8_rows=attempt8_split,
        output_path=plots_dir / "comparison_calibration_models_20pct_vs_attempt5.png",
    )

    text_lines = [
        "# Study02 Comparison vs Attempt 5",
        "",
        "Both studies use the same Nenrus rows and the same calibration protocol.",
        "",
        "## Overall MAE",
        "",
        "| Stage | Attempt 5 MAE | Attempt 8 MAE | Attempt 8 - Attempt 5 |",
        "|---|---:|---:|---:|",
    ]
    for row in overall_rows:
        text_lines.append(
            "| {} | {:.4f}m | {:.4f}m | {:+.4f}m |".format(
                str(row["stage"]),
                float(row["attempt5_mae"]),
                float(row["attempt8_mae"]),
                float(row["attempt8_minus_attempt5_mae"]),
            )
        )
    text_lines.extend(
        [
            "",
            "## By Drone Type",
            "",
            "| Drone | Attempt 5 raw | Attempt 8 raw | Attempt 5 calibrated | Attempt 8 calibrated |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in by_drone_rows:
        text_lines.append(
            "| {} | {:.4f}m | {:.4f}m | {:.4f}m | {:.4f}m |".format(
                str(row["drone_type"]),
                float(row["attempt5_raw_mae"]),
                float(row["attempt8_raw_mae"]),
                float(row["attempt5_calibrated_mae"]),
                float(row["attempt8_calibrated_mae"]),
            )
        )
    text_lines.extend(
        [
            "",
            "## Reading This Correctly",
            "",
            "The 20% split score is the cleaner estimate of how calibration generalizes when only part of the real dataset is used for calibration.",
            "The final full-data fit is useful as a best-fit mapping for this exact real dataset, but it is not a separate external-test score.",
            "",
        ]
    )
    _write_text(reports_dir / "comparison_vs_attempt5.md", "\n".join(text_lines))

    return {
        "reference_attempt": "attempt5_study04",
        "reference_fraction": reference_fraction,
        "overall_csv": str(reports_dir / "comparison_vs_attempt5.csv"),
        "by_drone_type_csv": str(reports_dir / "comparison_by_drone_type_vs_attempt5.csv"),
        "model_20pct_csv": str(reports_dir / "comparison_calibration_models_20pct_vs_attempt5.csv"),
        "analysis_md": str(reports_dir / "comparison_vs_attempt5.md"),
        "plots": [
            str(plots_dir / "comparison_overall_mae_vs_attempt5.png"),
            str(plots_dir / "comparison_by_drone_type_vs_attempt5.png"),
            str(plots_dir / "comparison_selected_calibration_fraction_track_vs_attempt5.png"),
            str(plots_dir / "comparison_calibration_models_20pct_vs_attempt5.png"),
        ],
        "overall_rows": overall_rows,
        "by_drone_type_rows": by_drone_rows,
        "model_20pct_rows": model_20_rows,
    }


def run(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _load_yaml(config_path)
    calibration_module = _load_attempt5_calibration_module()

    # Run the exact Attempt 5 calibration engine with the Attempt 8 config.
    summary = calibration_module.run(config_path=config_path)

    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    comparison = _build_comparison(config, summary, reports_dir, plots_dir)
    split_eval_summary_rows = _read_csv(reports_dir / "split_eval_summary.csv")
    mae_vs_rmse_plot_path = plots_dir / "mae_vs_rmse_calibration_summary.png"
    _plot_mae_vs_rmse(
        split_rows=split_eval_summary_rows,
        raw_metrics=dict(summary["raw_uncalibrated_metrics"]),
        final_metrics=dict(summary["final_metrics_full_data_fit"]),
        selected_model=str(summary["selected_model"]),
        output_path=mae_vs_rmse_plot_path,
    )

    summary_path = reports_dir / "summary.json"
    summary["study_name"] = str(config.get("study_name", "study02_nenrus_per_drone_calibration"))
    summary["attempt5_comparison"] = comparison
    summary["mae_vs_rmse_plot_png"] = str(mae_vs_rmse_plot_path)
    summary["plots"] = list(summary.get("plots", []))
    if str(mae_vs_rmse_plot_path) not in summary["plots"]:
        summary["plots"].append(str(mae_vs_rmse_plot_path))
    _write_json(summary_path, summary)

    print("[attempt8-study02] rows: {}".format(summary["num_rows"]))
    print("[attempt8-study02] raw MAE: {:.4f}m".format(float(summary["raw_uncalibrated_metrics"]["mae"])))
    print("[attempt8-study02] final calibrated MAE: {:.4f}m".format(float(summary["final_metrics_full_data_fit"]["mae"])))
    print("[attempt8-study02] wrote summary: {}".format(summary_path))
    return summary


def main() -> None:
    args = parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
