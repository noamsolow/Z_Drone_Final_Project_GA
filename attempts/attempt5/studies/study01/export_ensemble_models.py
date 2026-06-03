"""
Export fitted RF + XGBoost ensemble artifacts for attempt5.

This script intentionally trains only on the original Attempt 4 feature table.
It does not read or train on the external Nenrus dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt3.studies.study01.run_study_01 import (
    _fit_random_forest_estimator,
    _prediction_rows,
    _select_rows,
    _write_csv_rows,
    _write_json,
)
from attempts.attempt3.studies.study02.run_study_02 import (
    _build_global_categorical_levels,
    _build_rf_feature_names as _build_tabular_feature_names,
    _build_rf_matrix as _build_tabular_matrix,
)
from attempts.attempt4.studies.study02.run_study_02 import _fit_xgboost_estimator
from pipeline.calibration.evaluation import evaluate_predictions
from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export attempt5 RF+XGBoost ensemble models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the attempt5 export config.",
    )
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("YAML file did not contain an object: {}".format(path))
    return loaded


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("JSON file did not contain an object: {}".format(path))
    return loaded


def _load_feature_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _candidate_params_by_name(config: Dict[str, Any], section_name: str) -> Dict[str, Dict[str, Any]]:
    section = dict(config.get(section_name, {}))
    output: Dict[str, Dict[str, Any]] = {}
    for row in list(section.get("candidate_configs", [])):
        candidate = dict(row)
        name = str(candidate.pop("name"))
        if name in output:
            raise ValueError("Duplicate candidate '{}' in {}".format(name, section_name))
        output[name] = candidate
    if not output:
        raise ValueError("No candidates found in {}".format(section_name))
    return output


def _fit_models(
    rows: Sequence[Dict[str, Any]],
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Dict[str, Sequence[str]],
    rf_params: Dict[str, Any],
    xgb_params: Dict[str, Any],
) -> Dict[str, Any]:
    x = _build_tabular_matrix(
        rows,
        continuous_feature_names,
        categorical_feature_names,
        categorical_levels,
    )
    y = np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)

    rf_model = _fit_random_forest_estimator(rf_params)
    rf_model.fit(x, y)

    xgb_model = _fit_xgboost_estimator(xgb_params)
    xgb_model.fit(x, y)

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
    }


def _predict_ensemble(
    rf_model: Any,
    xgb_model: Any,
    rows: Sequence[Dict[str, Any]],
    continuous_feature_names: Sequence[str],
    categorical_feature_names: Sequence[str],
    categorical_levels: Dict[str, Sequence[str]],
    rf_weight: float,
    xgb_weight: float,
) -> Dict[str, np.ndarray]:
    x = _build_tabular_matrix(
        rows,
        continuous_feature_names,
        categorical_feature_names,
        categorical_levels,
    )
    rf_predictions = np.asarray(rf_model.predict(x), dtype=np.float64)
    xgb_predictions = np.asarray(xgb_model.predict(x), dtype=np.float64)
    ensemble_predictions = (rf_weight * rf_predictions) + (xgb_weight * xgb_predictions)
    return {
        "rf": rf_predictions,
        "xgb": xgb_predictions,
        "ensemble": ensemble_predictions,
    }


def _dump_scope_artifacts(
    scope_name: str,
    model_dir: Path,
    rf_model: Any,
    xgb_model: Any,
    schema: Dict[str, Any],
    blend_weights: Dict[str, float],
    train_row_count: int,
) -> Dict[str, str]:
    scope_dir = model_dir / scope_name
    scope_dir.mkdir(parents=True, exist_ok=True)

    rf_path = scope_dir / "rf_model.joblib"
    xgb_path = scope_dir / "xgb_model.joblib"
    ensemble_path = scope_dir / "ensemble.joblib"

    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(
        {
            "model_family": "rf_xgb_weighted_average",
            "scope": scope_name,
            "rf_model_path": str(rf_path),
            "xgb_model_path": str(xgb_path),
            "blend_weights": dict(blend_weights),
            "schema": dict(schema),
            "train_row_count": int(train_row_count),
            "load_note": "Load rf_model_path and xgb_model_path with joblib, then combine predictions using blend_weights.",
        },
        ensemble_path,
    )

    return {
        "rf_model_joblib": str(rf_path),
        "xgb_model_joblib": str(xgb_path),
        "ensemble_joblib": str(ensemble_path),
    }


def export_models(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = _load_yaml(config_path)

    sources = dict(config["sources"])
    feature_table_csv = _resolve_repo_path(str(sources["feature_table_csv"]))
    split_json = _resolve_repo_path(str(sources["split_json"]))
    rf_config_yaml = _resolve_repo_path(str(sources["attempt4_rf_config_yaml"]))
    xgb_config_yaml = _resolve_repo_path(str(sources["attempt4_xgb_config_yaml"]))

    rows = _load_feature_rows(feature_table_csv)
    split_payload = _load_json(split_json)
    rf_config = _load_yaml(rf_config_yaml)
    xgb_config = _load_yaml(xgb_config_yaml)

    model_config = dict(config["model"])
    raw_depth_feature_name = str(model_config["raw_depth_feature_name"])
    continuous_feature_names = [raw_depth_feature_name] + list(GEOMETRY_FEATURE_NAMES)
    categorical_feature_names = [str(name) for name in model_config["categorical_feature_names"]]
    categorical_levels = _build_global_categorical_levels(rows, categorical_feature_names)
    tabular_feature_names = _build_tabular_feature_names(
        continuous_feature_names,
        categorical_feature_names,
        categorical_levels,
    )

    rf_candidate_name = str(dict(model_config["rf"])["candidate_name"])
    xgb_candidate_name = str(dict(model_config["xgb"])["candidate_name"])
    rf_params = _candidate_params_by_name(rf_config, "upper_random_forest")[rf_candidate_name]
    xgb_params = _candidate_params_by_name(xgb_config, "upper_xgboost")[xgb_candidate_name]

    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    model_dir = output_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dev_rows = _select_rows(rows, split_payload["dev_indices"])
    test_rows = _select_rows(rows, split_payload["test_indices"])
    all_rows = list(rows)

    tuned_weights = {
        "rf_weight": float(dict(dict(model_config["blend"])["tuned"])["rf_weight"]),
        "xgb_weight": float(dict(dict(model_config["blend"])["tuned"])["xgb_weight"]),
    }
    equal_weights = {
        "rf_weight": float(dict(dict(model_config["blend"])["equal"])["rf_weight"]),
        "xgb_weight": float(dict(dict(model_config["blend"])["equal"])["xgb_weight"]),
    }

    schema = {
        "model_name": str(model_config["model_name"]),
        "continuous_feature_names": list(continuous_feature_names),
        "categorical_feature_names": list(categorical_feature_names),
        "categorical_levels": {key: list(value) for key, value in categorical_levels.items()},
        "tabular_feature_names": list(tabular_feature_names),
        "rf_candidate_name": rf_candidate_name,
        "xgb_candidate_name": xgb_candidate_name,
        "rf_params": dict(rf_params),
        "xgb_params": dict(xgb_params),
        "tuned_blend_weights": dict(tuned_weights),
        "equal_blend_weights": dict(equal_weights),
    }
    _write_json(schema, reports_dir / "model_schema.json")

    exported_scopes: Dict[str, Any] = {}

    if "dev_holdout" in list(config.get("export", {}).get("scopes", [])):
        print("[attempt5-study01] Fitting dev_holdout RF/XGB models | rows={}".format(len(dev_rows)))
        fitted = _fit_models(
            rows=dev_rows,
            continuous_feature_names=continuous_feature_names,
            categorical_feature_names=categorical_feature_names,
            categorical_levels=categorical_levels,
            rf_params=rf_params,
            xgb_params=xgb_params,
        )
        predictions = _predict_ensemble(
            rf_model=fitted["rf_model"],
            xgb_model=fitted["xgb_model"],
            rows=test_rows,
            continuous_feature_names=continuous_feature_names,
            categorical_feature_names=categorical_feature_names,
            categorical_levels=categorical_levels,
            rf_weight=tuned_weights["rf_weight"],
            xgb_weight=tuned_weights["xgb_weight"],
        )
        equal_predictions = (
            equal_weights["rf_weight"] * predictions["rf"]
            + equal_weights["xgb_weight"] * predictions["xgb"]
        )
        tuned_eval = evaluate_predictions(test_rows, predictions["ensemble"])
        equal_eval = evaluate_predictions(test_rows, equal_predictions)

        prediction_rows = _prediction_rows(
            rows=test_rows,
            predictions=predictions["ensemble"],
            model_name="attempt5_dev_holdout_tuned_weight_blend",
            split_name="test",
            model_family="rf_xgb_ensemble",
        )
        _write_csv_rows(prediction_rows, reports_dir / "dev_holdout_test_predictions.csv")
        _write_json(
            {
                "scope": "dev_holdout",
                "tuned_weight_blend": tuned_eval,
                "equal_weight_blend": equal_eval,
            },
            reports_dir / "dev_holdout_test_metrics.json",
        )

        paths = _dump_scope_artifacts(
            scope_name="dev_holdout",
            model_dir=model_dir,
            rf_model=fitted["rf_model"],
            xgb_model=fitted["xgb_model"],
            schema=schema,
            blend_weights=tuned_weights,
            train_row_count=len(dev_rows),
        )
        exported_scopes["dev_holdout"] = {
            "train_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "tuned_test_mae": float(tuned_eval["overall"]["mae"]),
            "tuned_test_mean_relative_error": float(tuned_eval["overall"]["mean_relative_error"]),
            "equal_test_mae": float(equal_eval["overall"]["mae"]),
            "equal_test_mean_relative_error": float(equal_eval["overall"]["mean_relative_error"]),
            "paths": paths,
        }

    if "full_dataset" in list(config.get("export", {}).get("scopes", [])):
        print("[attempt5-study01] Fitting full_dataset RF/XGB models | rows={}".format(len(all_rows)))
        fitted = _fit_models(
            rows=all_rows,
            continuous_feature_names=continuous_feature_names,
            categorical_feature_names=categorical_feature_names,
            categorical_levels=categorical_levels,
            rf_params=rf_params,
            xgb_params=xgb_params,
        )
        paths = _dump_scope_artifacts(
            scope_name="full_dataset",
            model_dir=model_dir,
            rf_model=fitted["rf_model"],
            xgb_model=fitted["xgb_model"],
            schema=schema,
            blend_weights=tuned_weights,
            train_row_count=len(all_rows),
        )
        exported_scopes["full_dataset"] = {
            "train_rows": len(all_rows),
            "paths": paths,
        }

    summary = {
        "study_name": str(config.get("study_name", "study01_export_rf_xgb_ensemble")),
        "feature_table_csv": str(feature_table_csv),
        "num_rows": len(rows),
        "num_dev_rows": len(dev_rows),
        "num_test_rows": len(test_rows),
        "model_name": str(model_config["model_name"]),
        "rf_candidate_name": rf_candidate_name,
        "xgb_candidate_name": xgb_candidate_name,
        "tuned_blend_weights": dict(tuned_weights),
        "equal_blend_weights": dict(equal_weights),
        "num_tabular_features": len(tabular_feature_names),
        "model_schema_json": str(reports_dir / "model_schema.json"),
        "exported_scopes": exported_scopes,
    }
    _write_json(summary, reports_dir / "summary.json")
    print("[attempt5-study01] Wrote summary: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    export_models(config_path=args.config)


if __name__ == "__main__":
    main()
