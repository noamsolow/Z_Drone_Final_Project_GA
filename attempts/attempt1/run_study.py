"""
Calibration study runner for the first attempt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.evaluation import (
    build_grouped_error_summary,
    compute_feature_correlations,
    evaluate_predictions,
)
from pipeline.calibration.features import read_regression_records_csv
from pipeline.calibration.linear_models import (
    fit_depth_geometry_metadata_linear_model,
    fit_depth_only_linear_model,
    fit_scale_only_model,
)
from pipeline.calibration.splits import make_distance_range_label, make_fixed_holdout_and_cv_splits


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_csv_rows(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_or_create_splits(
    records: Sequence[Any],
    split_dir: Path,
    holdout_fraction: float,
    num_folds: int,
    seed: int,
) -> Dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    test_split_path = split_dir / "test_split.json"
    cv_folds_path = split_dir / "cv_folds.json"

    if test_split_path.exists() and cv_folds_path.exists():
        test_split = json.loads(test_split_path.read_text(encoding="utf-8"))
        cv_folds = json.loads(cv_folds_path.read_text(encoding="utf-8"))
        return {
            "test_indices": test_split["test_indices"],
            "dev_indices": test_split["dev_indices"],
            "cv_folds": cv_folds["cv_folds"],
            "seed": test_split["seed"],
            "holdout_fraction": test_split["holdout_fraction"],
            "num_folds": cv_folds["num_folds"],
        }

    split_data = make_fixed_holdout_and_cv_splits(
        records=records,
        holdout_fraction=holdout_fraction,
        num_folds=num_folds,
        seed=seed,
    )

    test_split_payload = {
        "seed": split_data["seed"],
        "holdout_fraction": split_data["holdout_fraction"],
        "test_indices": split_data["test_indices"],
        "dev_indices": split_data["dev_indices"],
        "strata_by_index": split_data["strata_by_index"],
    }
    cv_folds_payload = {
        "seed": split_data["seed"],
        "num_folds": split_data["num_folds"],
        "cv_folds": split_data["cv_folds"],
    }
    test_split_path.write_text(json.dumps(test_split_payload, indent=2), encoding="utf-8")
    cv_folds_path.write_text(json.dumps(cv_folds_payload, indent=2), encoding="utf-8")

    return split_data


def _select_records(records: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [records[index] for index in indices]


def _prediction_rows(
    records: Sequence[Any],
    predictions: Sequence[float],
    model_name: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record, prediction in zip(records, predictions):
        true_distance = float(record.true_distance_m)
        absolute_error = abs(float(prediction) - true_distance)
        rows.append(
            {
                "model_name": model_name,
                "split_name": split_name,
                "image_path": record.image_path,
                "label_path": record.label_path,
                "weather": record.weather,
                "time_of_day": record.time_of_day,
                "distance_range": make_distance_range_label(true_distance),
                "true_distance_m": true_distance,
                "predicted_distance_m": float(prediction),
                "absolute_error_m": absolute_error,
                "relative_error": absolute_error / true_distance,
            }
        )
    return rows


def run_study(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_csv_path = output_root / "features" / "features.csv"
    if not features_csv_path.exists():
        raise FileNotFoundError(
            "Expected cached features at {}. Run first_attempt/run_extract_features.py first.".format(
                features_csv_path
            )
        )

    records = read_regression_records_csv(features_csv_path)
    split_config = resolved_config.get("split", {})
    split_data = _load_or_create_splits(
        records=records,
        split_dir=output_root / "splits",
        holdout_fraction=float(split_config.get("holdout_fraction", 0.15)),
        num_folds=int(split_config.get("num_folds", 5)),
        seed=int(split_config.get("seed", 0)),
    )

    fitters = {
        "scale_only": fit_scale_only_model,
        "depth_only_linear": fit_depth_only_linear_model,
        "depth_geometry_metadata_linear": fit_depth_geometry_metadata_linear_model,
    }

    dev_records = _select_records(records, split_data["dev_indices"])
    test_records = _select_records(records, split_data["test_indices"])

    cv_metrics: Dict[str, Any] = {"models": {}}
    cv_summary_rows: List[Dict[str, Any]] = []
    for model_name, fitter in fitters.items():
        fold_rows: List[Dict[str, Any]] = []
        for fold in split_data["cv_folds"]:
            train_records = _select_records(records, fold["train_indices"])
            val_records = _select_records(records, fold["val_indices"])
            model = fitter(train_records)
            predictions = model.predict(val_records)
            evaluation = evaluate_predictions(val_records, predictions)
            fold_row = {
                "fold_index": fold["fold_index"],
                "overall": evaluation["overall"],
                "per_distance_range": evaluation["per_distance_range"],
            }
            fold_rows.append(fold_row)

        mean_mae = sum(row["overall"]["mae"] for row in fold_rows) / len(fold_rows)
        mean_relative_error = sum(
            row["overall"]["mean_relative_error"] for row in fold_rows
        ) / len(fold_rows)
        cv_metrics["models"][model_name] = {
            "fold_metrics": fold_rows,
            "summary": {
                "mean_mae": mean_mae,
                "mean_relative_error": mean_relative_error,
            },
        }
        cv_summary_rows.append(
            {
                "model_name": model_name,
                "mean_mae": mean_mae,
                "mean_relative_error": mean_relative_error,
            }
        )

    ranked_models = sorted(
        cv_summary_rows,
        key=lambda row: (row["mean_mae"], row["mean_relative_error"], row["model_name"]),
    )
    winner_model_name = ranked_models[0]["model_name"]
    cv_metrics["winner_model_name"] = winner_model_name
    cv_metrics["ranking"] = ranked_models

    test_metrics: Dict[str, Any] = {
        "winner_model_name": winner_model_name,
        "models": {},
    }
    coefficient_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    grouped_summary_rows: List[Dict[str, Any]] = []

    for model_name, fitter in fitters.items():
        model = fitter(dev_records)
        predictions = model.predict(test_records)
        evaluation = evaluate_predictions(test_records, predictions)
        test_metrics["models"][model_name] = {
            "is_cv_winner": model_name == winner_model_name,
            "overall": evaluation["overall"],
            "per_distance_range": evaluation["per_distance_range"],
        }
        coefficient_rows.extend(model.coefficient_rows())
        prediction_rows.extend(_prediction_rows(test_records, predictions, model_name, "test"))
        grouped_summary_rows.extend(
            build_grouped_error_summary(
                records=test_records,
                predictions=predictions,
                model_name=model_name,
                split_name="test",
            )
        )

    correlation_rows = compute_feature_correlations(records)

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cv_metrics.json").write_text(json.dumps(cv_metrics, indent=2), encoding="utf-8")
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    _write_csv_rows(correlation_rows, reports_dir / "correlations.csv")
    _write_csv_rows(grouped_summary_rows, reports_dir / "grouped_error_summary.csv")
    _write_csv_rows(coefficient_rows, reports_dir / "model_coefficients.csv")
    _write_csv_rows(prediction_rows, reports_dir / "test_predictions.csv")

    return {
        "winner_model_name": winner_model_name,
        "cv_metrics_json": str(reports_dir / "cv_metrics.json"),
        "test_metrics_json": str(reports_dir / "test_metrics.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    result = run_study(config_path=args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
