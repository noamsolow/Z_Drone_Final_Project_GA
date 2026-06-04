"""
Attempt 6 Study 02: synthetic-only bbox scale domain randomization.

The real data is used only as an external test. The scale ranges are generic
training-time randomization settings, not fitted from the real dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES

from attempts.attempt6.studies.study01.run_study_01 import (
    _bucket_importance_rows,
    _categorical_levels,
    _depth_feature_names,
    _feature_importance_rows,
    _feature_names,
    _fit_rf,
    _fit_xgb,
    _group_metric_rows,
    _matrix,
    _metric_rows,
    _plot_bucket_importance,
    _plot_real_by_drone,
    _plot_real_overall,
    _plot_synthetic_vs_real,
    _plot_winner_true_vs_predicted,
    _prediction_rows,
    _read_csv_rows,
    _read_json,
    _read_yaml,
    _resolve_repo_path,
    _select_rows,
    _target_values,
    _variant_feature_names,
    _write_csv_rows,
    _write_json,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt6 Study 02 bbox scale randomization.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sample_scale(row_rng: np.random.Generator, scale_config: Dict[str, Any]) -> float:
    mode = str(scale_config.get("mode", "none"))
    if mode == "none":
        return 1.0
    if mode == "uniform":
        return float(
            row_rng.uniform(
                float(scale_config["min_scale"]),
                float(scale_config["max_scale"]),
            )
        )
    if mode == "lognormal":
        sampled = float(row_rng.lognormal(mean=0.0, sigma=float(scale_config["sigma"])))
        return float(
            np.clip(
                sampled,
                float(scale_config["min_scale"]),
                float(scale_config["max_scale"]),
            )
        )
    raise ValueError("Unsupported scale randomization mode '{}'.".format(mode))


def _scale_geometry_row(
    row: Dict[str, Any],
    *,
    scale: float,
    geometry_feature_names: Sequence[str],
) -> Dict[str, Any]:
    new_row = dict(row)
    if scale == 1.0:
        return new_row

    if "bbox_width_px" in geometry_feature_names:
        new_row["bbox_width_px"] = float(new_row["bbox_width_px"]) * scale
    if "bbox_width_norm" in geometry_feature_names:
        new_row["bbox_width_norm"] = float(new_row["bbox_width_norm"]) * scale
    if "bbox_height_px" in geometry_feature_names:
        new_row["bbox_height_px"] = float(new_row["bbox_height_px"]) * scale
    if "bbox_height_norm" in geometry_feature_names:
        new_row["bbox_height_norm"] = float(new_row["bbox_height_norm"]) * scale
    if "bbox_area_ratio" in geometry_feature_names:
        new_row["bbox_area_ratio"] = float(new_row["bbox_area_ratio"]) * scale * scale
    return new_row


def _scale_randomized_training_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    scale_config: Dict[str, Any],
    geometry_feature_names: Sequence[str],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: List[Dict[str, Any]] = []
    for row in rows:
        scale = _sample_scale(rng, scale_config)
        output.append(
            _scale_geometry_row(
                row,
                scale=scale,
                geometry_feature_names=geometry_feature_names,
            )
        )
    return output


def _plot_real_overprediction(metric_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    real_rows = [
        row for row in metric_rows
        if row["split_name"] == "real_external" and row["model_name"] == "blend"
    ]
    labels = [str(row["variant_name"]) for row in real_rows]
    values = [100.0 * float(row["overprediction_rate"]) for row in real_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, values, color="#b85c45")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Overprediction Rate (%)")
    ax.set_title("Real External Overprediction by Variant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_study(config_path: Optional[Path] = None, dry_run: bool = False) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _read_yaml(config_path)

    sources = dict(config["sources"])
    synthetic_path = _resolve_repo_path(str(sources["synthetic_feature_table_csv"]))
    split_path = _resolve_repo_path(str(sources["synthetic_split_json"]))
    real_path = _resolve_repo_path(str(sources["real_feature_table_csv"]))
    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"

    synthetic_rows = _read_csv_rows(synthetic_path)
    real_rows = _read_csv_rows(real_path)
    split_payload = _read_json(split_path)
    train_rows = _select_rows(synthetic_rows, split_payload["dev_indices"])
    synthetic_holdout_rows = _select_rows(synthetic_rows, split_payload["test_indices"])

    feature_config = dict(config["features"])
    raw_depth_feature_name = str(feature_config["raw_depth_feature_name"])
    all_depth_feature_names = _depth_feature_names(synthetic_rows)
    categorical_feature_names = [str(name) for name in feature_config["categorical_feature_names"]]
    categorical_levels = _categorical_levels(train_rows, categorical_feature_names)
    variants = [dict(row) for row in list(config["variants"])]

    real_columns = set(real_rows[0].keys()) if real_rows else set()
    supported_variants: List[Dict[str, Any]] = []
    skipped_variants: List[Dict[str, Any]] = []
    for variant in variants:
        feature_names = _variant_feature_names(
            variant,
            raw_depth_feature_name,
            all_depth_feature_names,
        )
        missing_real_features = [name for name in feature_names if name not in real_columns]
        if missing_real_features:
            skipped_variants.append(
                {
                    "name": str(variant["name"]),
                    "reason": "missing_real_features",
                    "missing_real_features": missing_real_features,
                }
            )
        else:
            supported_variants.append(variant)
    variants = supported_variants

    print("[attempt6-study02] Synthetic-only bbox scale domain randomization")
    print("  synthetic_train_rows: {}".format(len(train_rows)))
    print("  synthetic_holdout_rows: {}".format(len(synthetic_holdout_rows)))
    print("  real_external_rows: {}".format(len(real_rows)))
    print("  variants: {}".format(", ".join(str(row["name"]) for row in variants)))
    print("  real data use: external evaluation only")

    if dry_run:
        return {
            "study_name": str(config["study_name"]),
            "dry_run": True,
            "synthetic_train_rows": len(train_rows),
            "synthetic_holdout_rows": len(synthetic_holdout_rows),
            "real_external_rows": len(real_rows),
            "variant_names": [str(row["name"]) for row in variants],
            "skipped_variants": skipped_variants,
            "categorical_levels": categorical_levels,
        }

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_config = dict(config["model"])
    rf_params = dict(model_config["rf"])
    xgb_params = dict(model_config["xgb"])
    blend = dict(model_config["blend"])
    rf_weight = float(blend["rf_weight"])
    xgb_weight = float(blend["xgb_weight"])

    metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []

    for variant_index, variant in enumerate(variants, start=1):
        variant_name = str(variant["name"])
        continuous_feature_names = _variant_feature_names(
            variant,
            raw_depth_feature_name,
            all_depth_feature_names,
        )
        tabular_feature_names = _feature_names(
            continuous_feature_names,
            categorical_feature_names,
            categorical_levels,
        )
        geometry_feature_names = [name for name in continuous_feature_names if name in GEOMETRY_FEATURE_NAMES]
        scale_config = dict(variant.get("scale_randomization", {"mode": "none"}))

        print("[attempt6-study02] Variant {}/{} | {}".format(variant_index, len(variants), variant_name))
        train_variant_rows = _scale_randomized_training_rows(
            train_rows,
            scale_config=scale_config,
            geometry_feature_names=geometry_feature_names,
            seed=20260602 + variant_index,
        )

        x_train = _matrix(train_variant_rows, continuous_feature_names, categorical_feature_names, categorical_levels)
        y_train = _target_values(train_variant_rows)
        x_synthetic = _matrix(synthetic_holdout_rows, continuous_feature_names, categorical_feature_names, categorical_levels)
        x_real = _matrix(real_rows, continuous_feature_names, categorical_feature_names, categorical_levels)

        rf_model = _fit_rf(rf_params)
        rf_model.fit(x_train, y_train)
        xgb_model = _fit_xgb(xgb_params)
        xgb_model.fit(x_train, y_train)

        for split_name, rows_for_split, x_split in [
            ("synthetic_holdout", synthetic_holdout_rows, x_synthetic),
            ("real_external", real_rows, x_real),
        ]:
            rf_predictions = np.asarray(rf_model.predict(x_split), dtype=np.float64)
            xgb_predictions = np.asarray(xgb_model.predict(x_split), dtype=np.float64)
            blend_predictions = (rf_weight * rf_predictions) + (xgb_weight * xgb_predictions)

            for model_name, predictions in [
                ("rf", rf_predictions),
                ("xgb", xgb_predictions),
                ("blend", blend_predictions),
            ]:
                metric_rows.extend(_metric_rows(variant_name, model_name, split_name, rows_for_split, predictions))
                prediction_rows.extend(
                    _prediction_rows(
                        rows_for_split,
                        predictions,
                        variant_name=variant_name,
                        model_name=model_name,
                        split_name=split_name,
                    )
                )

        importance_rows.extend(
            _feature_importance_rows(
                variant_name=variant_name,
                feature_names=tabular_feature_names,
                rf_model=rf_model,
                xgb_model=xgb_model,
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
            )
        )
        real_blend = next(
            row for row in metric_rows
            if row["variant_name"] == variant_name and row["model_name"] == "blend" and row["split_name"] == "real_external"
        )
        print(
            "  real blend MAE={:.4f} | MRE={:.4f} | overprediction={:.4f}".format(
                float(real_blend["mae"]),
                float(real_blend["mean_relative_error"]),
                float(real_blend["overprediction_rate"]),
            )
        )

    real_predictions = [row for row in prediction_rows if row["split_name"] == "real_external"]
    real_by_drone = _group_metric_rows(real_predictions, ["drone_type"])
    real_by_distance = _group_metric_rows(real_predictions, ["true_distance_m"])
    bucket_rows = _bucket_importance_rows(importance_rows)

    _write_csv_rows(reports_dir / "variant_metrics.csv", metric_rows)
    _write_csv_rows(reports_dir / "predictions.csv", prediction_rows)
    _write_csv_rows(reports_dir / "real_by_drone_type.csv", real_by_drone)
    _write_csv_rows(reports_dir / "real_by_distance.csv", real_by_distance)
    _write_csv_rows(reports_dir / "feature_importances.csv", importance_rows)
    _write_csv_rows(reports_dir / "feature_bucket_importances.csv", bucket_rows)

    real_blend_rows = [
        row for row in metric_rows
        if row["split_name"] == "real_external" and row["model_name"] == "blend"
    ]
    synthetic_blend_rows = [
        row for row in metric_rows
        if row["split_name"] == "synthetic_holdout" and row["model_name"] == "blend"
    ]
    winner_row = min(real_blend_rows, key=lambda row: float(row["mae"]))
    winner_variant_name = str(winner_row["variant_name"])

    _plot_real_overall(metric_rows, plots_dir / "real_external_overall_by_variant.png")
    _plot_synthetic_vs_real(metric_rows, plots_dir / "synthetic_vs_real_mae_by_variant.png")
    _plot_real_by_drone(real_by_drone, plots_dir / "real_mae_by_drone_type_and_variant.png")
    _plot_bucket_importance(bucket_rows, plots_dir / "feature_bucket_importances_by_variant.png")
    _plot_real_overprediction(metric_rows, plots_dir / "real_overprediction_by_variant.png")
    _plot_winner_true_vs_predicted(real_predictions, winner_variant_name, plots_dir / "winner_real_true_vs_predicted.png")

    summary = {
        "study_name": str(config["study_name"]),
        "config_path": str(config_path.resolve()),
        "synthetic_feature_table_csv": str(synthetic_path),
        "real_feature_table_csv": str(real_path),
        "num_synthetic_train_rows": len(train_rows),
        "num_synthetic_holdout_rows": len(synthetic_holdout_rows),
        "num_real_external_rows": len(real_rows),
        "real_data_usage": "external evaluation only; no real rows are used for fitting or scale selection",
        "scale_randomization_note": "Scale ranges are generic domain-randomization settings, not fitted from Nenrus statistics.",
        "raw_depth_feature_name": raw_depth_feature_name,
        "categorical_levels_from_synthetic_train": categorical_levels,
        "skipped_variants": skipped_variants,
        "rf_weight": rf_weight,
        "xgb_weight": xgb_weight,
        "winner_by_real_blend_mae": winner_row,
        "synthetic_blend_metrics": synthetic_blend_rows,
        "real_blend_metrics": real_blend_rows,
        "reports": {
            "variant_metrics_csv": str(reports_dir / "variant_metrics.csv"),
            "predictions_csv": str(reports_dir / "predictions.csv"),
            "real_by_drone_type_csv": str(reports_dir / "real_by_drone_type.csv"),
            "real_by_distance_csv": str(reports_dir / "real_by_distance.csv"),
            "feature_importances_csv": str(reports_dir / "feature_importances.csv"),
            "feature_bucket_importances_csv": str(reports_dir / "feature_bucket_importances.csv"),
        },
        "plots": {
            "real_external_overall_by_variant_png": str(plots_dir / "real_external_overall_by_variant.png"),
            "synthetic_vs_real_mae_by_variant_png": str(plots_dir / "synthetic_vs_real_mae_by_variant.png"),
            "real_mae_by_drone_type_and_variant_png": str(plots_dir / "real_mae_by_drone_type_and_variant.png"),
            "feature_bucket_importances_by_variant_png": str(plots_dir / "feature_bucket_importances_by_variant.png"),
            "real_overprediction_by_variant_png": str(plots_dir / "real_overprediction_by_variant.png"),
            "winner_real_true_vs_predicted_png": str(plots_dir / "winner_real_true_vs_predicted.png"),
        },
    }
    _write_json(reports_dir / "summary.json", summary)

    analysis_md = "\n".join(
        [
            "# Attempt6 Study02 Analysis",
            "",
            "This study trains only on synthetic rows with generic bbox scale domain randomization.",
            "The real Nenrus rows are used only as an external test.",
            "",
            "## Winner on Real External Blend MAE",
            "",
            "- Variant: `{}`".format(winner_variant_name),
            "- Real MAE: `{:.4f}m`".format(float(winner_row["mae"])),
            "- Real mean relative error: `{:.4f}`".format(float(winner_row["mean_relative_error"])),
            "- Real overprediction rate: `{:.4f}`".format(float(winner_row["overprediction_rate"])),
        ]
    )
    (reports_dir / "analysis.md").write_text(analysis_md, encoding="utf-8")

    print("[attempt6-study02] Study complete")
    print("  winner_variant: {}".format(winner_variant_name))
    print("  winner_real_mae: {:.4f}".format(float(winner_row["mae"])))
    print("  summary_json: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    result = run_study(config_path=args.config, dry_run=bool(args.dry_run))
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
