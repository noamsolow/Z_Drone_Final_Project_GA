"""
Study 03 for attempt4: blend the best random-forest and XGBoost models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study01.run_study_01 import _write_csv_rows, _write_json
from attempts.attempt3.studies.study03.run_study_03 import _resolve_repo_path
from pipeline.calibration.evaluation import build_grouped_error_summary, evaluate_predictions
from pipeline.calibration.splits import make_distance_range_label


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attempt4 Study 03 RF/XGBoost blend study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study 03 YAML config.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv_rows(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError("Expected CSV does not exist: {}".format(input_path))
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_json(input_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError("Expected JSON does not exist: {}".format(input_path))
    return json.loads(input_path.read_text(encoding="utf-8"))


def _winner_model_key(summary_payload: Dict[str, Any]) -> str:
    return "{}__{}".format(
        str(summary_payload["winner_model_name"]),
        str(summary_payload["winner_candidate_name"]),
    )


def _filter_prediction_rows(
    rows: Sequence[Dict[str, Any]],
    model_key: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    filtered = [
        dict(row)
        for row in rows
        if str(row.get("model_name", "")) == model_key and str(row.get("split_name", "")) == split_name
    ]
    if not filtered:
        raise ValueError(
            "No prediction rows found for model '{}' and split '{}'.".format(model_key, split_name)
        )
    return filtered


def _parse_prediction_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(raw_row)
    for key in [
        "true_distance_m",
        "predicted_distance_m",
        "signed_error_m",
        "absolute_error_m",
        "relative_error",
    ]:
        row[key] = float(row[key])
    depth_lower_value = str(row.get("depth_lower_model_prediction", "")).strip()
    row["depth_lower_model_prediction"] = float(depth_lower_value) if depth_lower_value else None
    return row


def _merge_prediction_pairs(
    rf_rows: Sequence[Dict[str, Any]],
    xgb_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rf_by_id = {_parse_prediction_row(row)["augmented_row_id"]: _parse_prediction_row(row) for row in rf_rows}
    xgb_by_id = {_parse_prediction_row(row)["augmented_row_id"]: _parse_prediction_row(row) for row in xgb_rows}
    if set(rf_by_id.keys()) != set(xgb_by_id.keys()):
        missing_from_xgb = sorted(set(rf_by_id.keys()) - set(xgb_by_id.keys()))
        missing_from_rf = sorted(set(xgb_by_id.keys()) - set(rf_by_id.keys()))
        raise ValueError(
            "RF/XGB prediction ids do not match. Missing from XGB: {} | Missing from RF: {}".format(
                missing_from_xgb[:10],
                missing_from_rf[:10],
            )
        )

    merged_rows: List[Dict[str, Any]] = []
    for augmented_row_id in sorted(rf_by_id.keys()):
        rf_row = rf_by_id[augmented_row_id]
        xgb_row = xgb_by_id[augmented_row_id]
        for key in [
            "true_distance_m",
            "weather",
            "time_of_day",
            "distance_range",
            "source_image_path",
            "source_label_path",
            "jitter_name",
        ]:
            if str(rf_row.get(key, "")) != str(xgb_row.get(key, "")):
                raise ValueError(
                    "RF/XGB prediction rows disagree on '{}' for '{}'.".format(key, augmented_row_id)
                )
        if abs(float(rf_row["true_distance_m"]) - float(xgb_row["true_distance_m"])) > 1e-9:
            raise ValueError("RF/XGB true distances differ for '{}'.".format(augmented_row_id))

        merged_rows.append(
            {
                "augmented_row_id": str(augmented_row_id),
                "image_path": str(rf_row["image_path"]),
                "label_path": str(rf_row["label_path"]),
                "source_image_path": str(rf_row.get("source_image_path", "")),
                "source_label_path": str(rf_row.get("source_label_path", "")),
                "jitter_name": str(rf_row.get("jitter_name", "")),
                "weather": str(rf_row["weather"]),
                "time_of_day": str(rf_row["time_of_day"]),
                "distance_range": str(rf_row["distance_range"]),
                "true_distance_m": float(rf_row["true_distance_m"]),
                "rf_prediction_m": float(rf_row["predicted_distance_m"]),
                "xgb_prediction_m": float(xgb_row["predicted_distance_m"]),
            }
        )
    return merged_rows


def _prediction_rows_for_blend(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[float],
    model_name: str,
    split_name: str,
    rf_weight: float,
    xgb_weight: float,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        true_distance = float(row["true_distance_m"])
        signed_error = float(prediction - true_distance)
        output_rows.append(
            {
                "model_name": model_name,
                "model_family": "blend",
                "split_name": split_name,
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "source_image_path": row["source_image_path"],
                "source_label_path": row["source_label_path"],
                "jitter_name": row["jitter_name"],
                "augmented_row_id": row["augmented_row_id"],
                "weather": row["weather"],
                "time_of_day": row["time_of_day"],
                "distance_range": make_distance_range_label(true_distance),
                "true_distance_m": true_distance,
                "predicted_distance_m": float(prediction),
                "rf_prediction_m": float(row["rf_prediction_m"]),
                "xgb_prediction_m": float(row["xgb_prediction_m"]),
                "rf_weight": float(rf_weight),
                "xgb_weight": float(xgb_weight),
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / true_distance if true_distance > 0 else 0.0,
            }
        )
    return output_rows


def _blend_predictions(
    rows: Sequence[Dict[str, Any]],
    rf_weight: float,
    clip_min_distance_m: float,
) -> np.ndarray:
    rf_values = np.asarray([float(row["rf_prediction_m"]) for row in rows], dtype=np.float64)
    xgb_values = np.asarray([float(row["xgb_prediction_m"]) for row in rows], dtype=np.float64)
    predictions = rf_weight * rf_values + (1.0 - rf_weight) * xgb_values
    if clip_min_distance_m is not None:
        predictions = np.maximum(predictions, float(clip_min_distance_m))
    return predictions


def _evaluate_model(
    rows: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
) -> Dict[str, Any]:
    eval_payload = evaluate_predictions(rows, predictions)
    return {
        "overall": dict(eval_payload["overall"]),
        "per_distance_range": dict(eval_payload["per_distance_range"]),
    }


def _blend_search_rows(
    dev_rows: Sequence[Dict[str, Any]],
    weight_grid_step: float,
    clip_min_distance_m: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if weight_grid_step <= 0.0 or weight_grid_step > 1.0:
        raise ValueError("weight_grid_step must be in (0, 1].")
    steps = int(round(1.0 / weight_grid_step))
    candidate_weights = sorted({round(index * weight_grid_step, 10) for index in range(steps + 1)} | {1.0})
    output_rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    for rf_weight in candidate_weights:
        predictions = _blend_predictions(
            dev_rows,
            rf_weight=float(rf_weight),
            clip_min_distance_m=clip_min_distance_m,
        )
        evaluation = evaluate_predictions(dev_rows, predictions)["overall"]
        row = {
            "rf_weight": float(rf_weight),
            "xgb_weight": float(1.0 - rf_weight),
            "oof_mae": float(evaluation["mae"]),
            "oof_mean_relative_error": float(evaluation["mean_relative_error"]),
            "oof_rmse": float(evaluation["rmse"]),
            "oof_r2": float(evaluation["r2"]),
        }
        output_rows.append(row)
        if best_row is None or (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            float(row["rf_weight"]),
        ) < (
            float(best_row["oof_mae"]),
            float(best_row["oof_mean_relative_error"]),
            float(best_row["rf_weight"]),
        ):
            best_row = row
    if best_row is None:
        raise ValueError("Blend search did not produce any candidate rows.")
    return output_rows, best_row


def run_study(config_path: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    resolved_config = load_config(config_path)
    output_root = _resolve_repo_path(str(resolved_config["output_root"]))
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    sources = dict(resolved_config.get("sources", {}))
    rf_summary_json = _resolve_repo_path(str(sources["rf_summary_json"]))
    rf_cv_predictions_csv = _resolve_repo_path(str(sources["rf_cv_predictions_csv"]))
    rf_test_predictions_csv = _resolve_repo_path(str(sources["rf_test_predictions_csv"]))
    xgb_summary_json = _resolve_repo_path(str(sources["xgb_summary_json"]))
    xgb_cv_predictions_csv = _resolve_repo_path(str(sources["xgb_cv_predictions_csv"]))
    xgb_test_predictions_csv = _resolve_repo_path(str(sources["xgb_test_predictions_csv"]))

    rf_summary = _load_json(rf_summary_json)
    xgb_summary = _load_json(xgb_summary_json)
    rf_model_key = _winner_model_key(rf_summary)
    xgb_model_key = _winner_model_key(xgb_summary)

    rf_cv_rows = _filter_prediction_rows(_read_csv_rows(rf_cv_predictions_csv), rf_model_key, "cv_oof")
    rf_test_rows = _filter_prediction_rows(_read_csv_rows(rf_test_predictions_csv), rf_model_key, "test")
    xgb_cv_rows = _filter_prediction_rows(_read_csv_rows(xgb_cv_predictions_csv), xgb_model_key, "cv_oof")
    xgb_test_rows = _filter_prediction_rows(_read_csv_rows(xgb_test_predictions_csv), xgb_model_key, "test")

    dev_rows = _merge_prediction_pairs(rf_cv_rows, xgb_cv_rows)
    test_rows = _merge_prediction_pairs(rf_test_rows, xgb_test_rows)

    blend_config = dict(resolved_config.get("blend", {}))
    weight_grid_step = float(blend_config.get("weight_grid_step", 0.01))
    clip_min_distance_m = float(blend_config.get("clip_min_distance_m", 0.0))

    if verbose:
        print(
            "[attempt4-study03] RF/XGB blend study\n"
            "  config: {}\n"
            "  rf_winner: {}\n"
            "  xgb_winner: {}\n"
            "  dev_rows: {}\n"
            "  test_rows: {}\n"
            "  weight_grid_step: {}".format(
                config_path,
                rf_model_key,
                xgb_model_key,
                len(dev_rows),
                len(test_rows),
                weight_grid_step,
            )
        )

    search_rows, best_weight_row = _blend_search_rows(
        dev_rows=dev_rows,
        weight_grid_step=weight_grid_step,
        clip_min_distance_m=clip_min_distance_m,
    )
    tuned_rf_weight = float(best_weight_row["rf_weight"])
    tuned_xgb_weight = float(best_weight_row["xgb_weight"])

    model_specs = [
        {"model_name": "rf_winner_baseline", "rf_weight": 1.0, "xgb_weight": 0.0},
        {"model_name": "xgb_winner_baseline", "rf_weight": 0.0, "xgb_weight": 1.0},
        {"model_name": "equal_weight_blend", "rf_weight": 0.5, "xgb_weight": 0.5},
        {"model_name": "tuned_weight_blend", "rf_weight": tuned_rf_weight, "xgb_weight": tuned_xgb_weight},
    ]

    cv_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "rf_winner_model_key": rf_model_key,
        "xgb_winner_model_key": xgb_model_key,
        "weight_grid_step": weight_grid_step,
        "blend_search_best_row": dict(best_weight_row),
        "models": {},
    }
    test_metrics: Dict[str, Any] = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "rf_winner_model_key": rf_model_key,
        "xgb_winner_model_key": xgb_model_key,
        "models": {},
    }

    cv_prediction_rows: List[Dict[str, Any]] = []
    test_prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []

    for model_spec in model_specs:
        model_name = str(model_spec["model_name"])
        rf_weight = float(model_spec["rf_weight"])
        xgb_weight = float(model_spec["xgb_weight"])
        dev_predictions = _blend_predictions(dev_rows, rf_weight=rf_weight, clip_min_distance_m=clip_min_distance_m)
        test_predictions = _blend_predictions(test_rows, rf_weight=rf_weight, clip_min_distance_m=clip_min_distance_m)
        dev_eval = _evaluate_model(dev_rows, dev_predictions)
        test_eval = _evaluate_model(test_rows, test_predictions)

        cv_metrics["models"][model_name] = {
            "rf_weight": rf_weight,
            "xgb_weight": xgb_weight,
            "overall": dict(dev_eval["overall"]),
            "per_distance_range": dict(dev_eval["per_distance_range"]),
        }
        test_metrics["models"][model_name] = {
            "rf_weight": rf_weight,
            "xgb_weight": xgb_weight,
            "overall": dict(test_eval["overall"]),
            "per_distance_range": dict(test_eval["per_distance_range"]),
        }
        ranking_rows.append(
            {
                "model_name": model_name,
                "rf_weight": rf_weight,
                "xgb_weight": xgb_weight,
                "oof_mae": float(dev_eval["overall"]["mae"]),
                "oof_mean_relative_error": float(dev_eval["overall"]["mean_relative_error"]),
            }
        )
        cv_prediction_rows.extend(
            _prediction_rows_for_blend(
                rows=dev_rows,
                predictions=dev_predictions,
                model_name=model_name,
                split_name="cv_oof",
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
            )
        )
        test_prediction_rows.extend(
            _prediction_rows_for_blend(
                rows=test_rows,
                predictions=test_predictions,
                model_name=model_name,
                split_name="test",
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=dev_rows,
                predictions=dev_predictions,
                model_name=model_name,
                split_name="cv_oof",
            )
        )
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=test_rows,
                predictions=test_predictions,
                model_name=model_name,
                split_name="test",
            )
        )
        if verbose:
            print(
                "[attempt4-study03] Evaluated {} | rf_weight={:.2f} | xgb_weight={:.2f} | cv_mae={:.4f} | test_mae={:.4f}".format(
                    model_name,
                    rf_weight,
                    xgb_weight,
                    float(dev_eval["overall"]["mae"]),
                    float(test_eval["overall"]["mae"]),
                )
            )

    ranked_models = sorted(
        ranking_rows,
        key=lambda row: (
            float(row["oof_mae"]),
            float(row["oof_mean_relative_error"]),
            str(row["model_name"]),
        ),
    )
    for rank_index, row in enumerate(ranked_models, start=1):
        row["rank"] = rank_index
    winner_model_name = str(ranked_models[0]["model_name"])
    winner_test_metrics = test_metrics["models"][winner_model_name]["overall"]
    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["ranking"] = ranked_models
    test_metrics["winner_model_name"] = winner_model_name

    _write_json(cv_metrics, reports_dir / "cv_metrics.json")
    _write_json(test_metrics, reports_dir / "test_metrics.json")
    _write_csv_rows(search_rows, reports_dir / "blend_search.csv")
    _write_csv_rows(cv_prediction_rows, reports_dir / "cv_predictions.csv")
    _write_csv_rows(test_prediction_rows, reports_dir / "test_predictions.csv")
    _write_csv_rows(grouped_summary_rows, reports_dir / "grouped_error_summary.csv")

    summary_payload = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "rf_winner_model_key": rf_model_key,
        "xgb_winner_model_key": xgb_model_key,
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "weight_grid_step": weight_grid_step,
        "best_blend_search_row": dict(best_weight_row),
        "winner_model_name": winner_model_name,
        "winner_cv_oof_mae": float(ranked_models[0]["oof_mae"]),
        "winner_cv_oof_mean_relative_error": float(ranked_models[0]["oof_mean_relative_error"]),
        "winner_test_mae": float(winner_test_metrics["mae"]),
        "winner_test_mean_relative_error": float(winner_test_metrics["mean_relative_error"]),
        "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
        "test_metrics_json": str(reports_dir / "test_metrics.json"),
        "blend_search_csv": str(reports_dir / "blend_search.csv"),
    }
    _write_json(summary_payload, reports_dir / "summary.json")

    if verbose:
        print(
            "[attempt4-study03] Study complete\n"
            "  winner_model: {}\n"
            "  winner_test_mae: {:.4f}\n"
            "  summary_json: {}".format(
                winner_model_name,
                float(winner_test_metrics["mae"]),
                reports_dir / "summary.json",
            )
        )
    return summary_payload


def main() -> None:
    args = parse_args()
    run_study(config_path=args.config, verbose=True)


if __name__ == "__main__":
    main()
