"""
Train depth-weighted variants on the original data and evaluate on Nenrus.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study01.run_study_01 import _fit_random_forest_estimator
from attempts.attempt4.studies.study02.run_study_02 import _fit_xgboost_estimator


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
GEOMETRY_FEATURES = [
    "bbox_width_px",
    "bbox_height_px",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
]
SIZE_GEOMETRY_FEATURES = [
    "bbox_width_px",
    "bbox_height_px",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_ratio",
]
NO_SIZE_GEOMETRY_FEATURES = [
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
]
NORMALIZED_GEOMETRY_FEATURES = [
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
]
CATEGORICAL_FEATURES = ["weather", "time_of_day"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run depth-weighted external variants.")
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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON did not contain an object: {}".format(path))
    return payload


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_categorical_levels(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {
        feature_name: sorted({str(row[feature_name]) for row in rows})
        for feature_name in CATEGORICAL_FEATURES
    }


def _continuous_specs_for_variant(raw_depth_name: str, variant_name: str) -> List[Tuple[str, str]]:
    if variant_name == "depth_only":
        return [("raw_depth", raw_depth_name)]
    if variant_name == "depth_plus_no_size_geometry":
        return [("raw_depth", raw_depth_name)] + [(name, name) for name in NO_SIZE_GEOMETRY_FEATURES]
    if variant_name == "depth_plus_normalized_geometry":
        return [("raw_depth", raw_depth_name)] + [(name, name) for name in NORMALIZED_GEOMETRY_FEATURES]
    if variant_name == "depth_repeated_8_with_geometry":
        return [("raw_depth_dup_{:02d}".format(i), raw_depth_name) for i in range(8)] + [(name, name) for name in GEOMETRY_FEATURES]
    if variant_name == "depth_repeated_16_with_geometry":
        return [("raw_depth_dup_{:02d}".format(i), raw_depth_name) for i in range(16)] + [(name, name) for name in GEOMETRY_FEATURES]
    raise ValueError("Unsupported variant: {}".format(variant_name))


def _feature_names(
    continuous_specs: Sequence[Tuple[str, str]],
    categorical_levels: Dict[str, Sequence[str]],
    include_categoricals: bool,
) -> List[str]:
    names = [alias for alias, _ in continuous_specs]
    if include_categoricals:
        for feature_name in CATEGORICAL_FEATURES:
            for level in categorical_levels.get(feature_name, []):
                names.append("{}={}".format(feature_name, level))
    return names


def _build_matrix(
    rows: Sequence[Dict[str, Any]],
    continuous_specs: Sequence[Tuple[str, str]],
    categorical_levels: Dict[str, Sequence[str]],
    include_categoricals: bool,
) -> np.ndarray:
    matrix_rows: List[List[float]] = []
    for row in rows:
        values = [float(row[source_name]) for _, source_name in continuous_specs]
        if include_categoricals:
            for feature_name in CATEGORICAL_FEATURES:
                row_value = str(row.get(feature_name, "unknown"))
                for level in categorical_levels.get(feature_name, []):
                    values.append(1.0 if row_value == level else 0.0)
        matrix_rows.append(values)
    return np.asarray(matrix_rows, dtype=np.float64)


def _metrics(rows: Sequence[Dict[str, Any]], predictions: Sequence[float]) -> Dict[str, Any]:
    true_values = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    errors = preds - true_values
    abs_errors = np.abs(errors)
    rel_errors = np.divide(abs_errors, true_values, out=np.zeros_like(abs_errors), where=true_values > 0)
    centered = true_values - float(np.mean(true_values))
    sst = float(np.sum(centered**2))
    sse = float(np.sum(errors**2))
    return {
        "count": int(len(rows)),
        "mae": float(np.mean(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1.0 - (sse / sst)) if sst > 0 else None,
        "mean_signed_error": float(np.mean(errors)),
        "median_signed_error": float(np.median(errors)),
        "p90_absolute_error": float(np.percentile(abs_errors, 90)),
        "p95_absolute_error": float(np.percentile(abs_errors, 95)),
        "within_5m_rate": float(np.mean(abs_errors <= 5.0)),
        "within_10m_rate": float(np.mean(abs_errors <= 10.0)),
        "within_20m_rate": float(np.mean(abs_errors <= 20.0)),
        "overprediction_rate": float(np.mean(errors > 0)),
        "underprediction_rate": float(np.mean(errors < 0)),
    }


def _prediction_rows(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    variant_name: str,
    model_family: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        signed_error = float(prediction - true_distance)
        output.append(
            {
                "variant_name": variant_name,
                "model_family": model_family,
                "sample_id": row.get("sample_id", ""),
                "drone_type": row.get("drone_type", "unknown"),
                "true_distance_m": true_distance,
                "predicted_distance_m": float(prediction),
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
            }
        )
    return output


def _group_metrics(prediction_rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output: List[Dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: item[0]):
        metric_row = {key: value for key, value in zip(keys, key_values)}
        metric_row.update(
            {
                "count": len(group),
                "mae": float(np.mean([float(row["absolute_error_m"]) for row in group])),
                "mean_relative_error": float(np.mean([float(row["relative_error"]) for row in group])),
                "mean_signed_error": float(np.mean([float(row["signed_error_m"]) for row in group])),
                "within_10m_rate": float(np.mean([float(row["absolute_error_m"]) <= 10.0 for row in group])),
            }
        )
        output.append(metric_row)
    return output


def _plot_model_comparison(metric_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ensemble_rows = [
        row for row in metric_rows
        if str(row["model_family"]) == "ensemble"
    ]
    ensemble_rows = sorted(ensemble_rows, key=lambda row: float(row["mae"]))
    labels = [str(row["variant_name"]) for row in ensemble_rows]
    mae_values = [float(row["mae"]) for row in ensemble_rows]
    rel_values = [float(row["mean_relative_error"]) for row in ensemble_rows]
    signed_values = [float(row["mean_signed_error"]) for row in ensemble_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(labels, mae_values, color="#365f46")
    axes[0].set_title("Nenrus MAE by Depth-Weighted Variant")
    axes[0].set_ylabel("MAE (m)")
    axes[1].bar(labels, rel_values, color="#4c6b8a")
    axes[1].set_title("Mean Relative Error")
    axes[1].set_ylabel("Relative error")
    axes[2].bar(labels, signed_values, color="#8a5135")
    axes[2].set_title("Mean Signed Error")
    axes[2].set_ylabel("Prediction - GT (m)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_by_distance(prediction_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    selected_variants = [
        "baseline_saved_ensemble",
        "depth_only",
        "depth_plus_no_size_geometry",
        "depth_repeated_16_with_geometry",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in selected_variants:
        subset = [row for row in prediction_rows if row["variant_name"] == variant and row["model_family"] == "ensemble"]
        grouped: Dict[float, List[float]] = defaultdict(list)
        for row in subset:
            grouped[float(row["true_distance_m"])].append(float(row["absolute_error_m"]))
        distances = sorted(grouped)
        values = [float(np.mean(grouped[distance])) for distance in distances]
        ax.plot(distances, values, marker="o", linewidth=2.0, label=variant)
    ax.set_title("MAE by Distance for Selected Depth-Weighted Variants")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("MAE (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_analysis(path: Path, metric_rows: Sequence[Dict[str, Any]]) -> None:
    ensemble_rows = sorted(
        [row for row in metric_rows if str(row["model_family"]) == "ensemble"],
        key=lambda row: float(row["mae"]),
    )
    lines = [
        "# Depth-Weighted External Variants Analysis",
        "",
        "## Question",
        "",
        "Can we reduce the Nenrus geometry-domain failure by making relative depth more influential?",
        "",
        "## Ensemble Results",
        "",
    ]
    for row in ensemble_rows:
        lines.append(
            "- `{}`: MAE `{:.3f}m`, mean relative error `{:.3f}`, mean signed error `{:.3f}m`, within 10m `{:.3f}`".format(
                row["variant_name"],
                float(row["mae"]),
                float(row["mean_relative_error"]),
                float(row["mean_signed_error"]),
                float(row["within_10m_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The best result came from `depth_plus_normalized_geometry`, not from `depth_only` and not from simply duplicating the depth feature.",
            "That means relative depth helps only when it is paired with some geometry, and the more important change was removing pixel-size geometry features that are especially domain-sensitive.",
            "",
            "`depth_only` performed much worse than the saved baseline, so relative depth by itself is not strong enough to solve the external-domain problem.",
            "Repeating the depth feature 8x or 16x also did not improve over the saved baseline, which suggests that simply forcing more feature-sampling attention onto depth is not the right fix.",
            "",
            "The best variant still overpredicted every Nenrus sample, so the core issue remains a systematic domain calibration problem.",
            "Depth weighting can reduce the error somewhat, but it does not replace real-domain calibration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _load_yaml(config_path)
    sources = dict(config["sources"])
    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    original_rows = _read_csv(_resolve_repo_path(str(sources["original_feature_table_csv"])))
    nenrus_rows = _read_csv(_resolve_repo_path(str(sources["nenrus_feature_table_csv"])))
    baseline_rows = _read_csv(_resolve_repo_path(str(sources["baseline_predictions_csv"])))
    schema = _load_json(_resolve_repo_path(str(sources["model_schema_json"])))

    raw_depth_name = str(config["model"]["raw_depth_feature_name"])
    categorical_levels = _build_categorical_levels(original_rows)
    rf_params = dict(schema["rf_params"])
    xgb_params = dict(schema["xgb_params"])
    rf_weight = float(config["model"]["blend"]["rf_weight"])
    xgb_weight = float(config["model"]["blend"]["xgb_weight"])

    variant_names = [
        "depth_only",
        "depth_plus_no_size_geometry",
        "depth_plus_normalized_geometry",
        "depth_repeated_8_with_geometry",
        "depth_repeated_16_with_geometry",
    ]

    all_prediction_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    baseline_predictions = [float(row["predicted_distance_m"]) for row in baseline_rows]
    baseline_metrics = _metrics(baseline_rows, baseline_predictions)
    baseline_metric_row = {
        "variant_name": "baseline_saved_ensemble",
        "model_family": "ensemble",
        "num_features": len(schema["tabular_feature_names"]),
    }
    baseline_metric_row.update(baseline_metrics)
    metric_rows.append(baseline_metric_row)
    all_prediction_rows.extend(
        _prediction_rows(
            baseline_rows,
            baseline_predictions,
            "baseline_saved_ensemble",
            "ensemble",
        )
    )

    for variant_index, variant_name in enumerate(variant_names, start=1):
        continuous_specs = _continuous_specs_for_variant(raw_depth_name, variant_name)
        include_categoricals = variant_name != "depth_only"
        feature_names = _feature_names(continuous_specs, categorical_levels, include_categoricals)
        print(
            "[attempt5-study03] Training variant {}/{} | {} | features={}".format(
                variant_index,
                len(variant_names),
                variant_name,
                len(feature_names),
            )
        )
        x_train = _build_matrix(original_rows, continuous_specs, categorical_levels, include_categoricals)
        y_train = np.asarray([float(row["true_distance_m"]) for row in original_rows], dtype=np.float64)
        x_external = _build_matrix(nenrus_rows, continuous_specs, categorical_levels, include_categoricals)

        rf_model = _fit_random_forest_estimator(rf_params)
        rf_model.fit(x_train, y_train)
        rf_predictions = np.asarray(rf_model.predict(x_external), dtype=np.float64)

        xgb_model = _fit_xgboost_estimator(xgb_params)
        xgb_model.fit(x_train, y_train)
        xgb_predictions = np.asarray(xgb_model.predict(x_external), dtype=np.float64)

        ensemble_predictions = (rf_weight * rf_predictions) + (xgb_weight * xgb_predictions)
        family_predictions = {
            "rf": rf_predictions,
            "xgb": xgb_predictions,
            "ensemble": ensemble_predictions,
        }
        for family_name, predictions in family_predictions.items():
            metric_row = {
                "variant_name": variant_name,
                "model_family": family_name,
                "num_features": len(feature_names),
                "feature_names": " | ".join(feature_names),
            }
            metric_row.update(_metrics(nenrus_rows, predictions))
            metric_rows.append(metric_row)
            all_prediction_rows.extend(_prediction_rows(nenrus_rows, predictions, variant_name, family_name))

    _write_csv(reports_dir / "model_metrics.csv", metric_rows)
    _write_csv(reports_dir / "predictions.csv", all_prediction_rows)
    _write_csv(
        reports_dir / "by_variant_drone_type.csv",
        _group_metrics(all_prediction_rows, ["variant_name", "model_family", "drone_type"]),
    )
    _write_csv(
        reports_dir / "by_variant_distance.csv",
        _group_metrics(all_prediction_rows, ["variant_name", "model_family", "true_distance_m"]),
    )
    _plot_model_comparison(metric_rows, plots_dir / "depth_weighted_variant_comparison.png")
    _plot_by_distance(all_prediction_rows, plots_dir / "depth_weighted_selected_variants_by_distance.png")
    _write_analysis(reports_dir / "analysis.md", metric_rows)

    best_ensemble = min(
        [row for row in metric_rows if row["model_family"] == "ensemble"],
        key=lambda row: float(row["mae"]),
    )
    summary = {
        "study_name": str(config.get("study_name", "study03_depth_weighted_external_variants")),
        "num_original_rows": len(original_rows),
        "num_nenrus_rows": len(nenrus_rows),
        "best_ensemble_variant": best_ensemble,
        "model_metrics_csv": str(reports_dir / "model_metrics.csv"),
        "predictions_csv": str(reports_dir / "predictions.csv"),
        "analysis_md": str(reports_dir / "analysis.md"),
        "plots": [
            str(plots_dir / "depth_weighted_variant_comparison.png"),
            str(plots_dir / "depth_weighted_selected_variants_by_distance.png"),
        ],
    }
    _write_json(reports_dir / "summary.json", summary)
    print("[attempt5-study03] Wrote summary: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
