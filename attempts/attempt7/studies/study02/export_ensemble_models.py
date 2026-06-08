"""
Export fitted RF + XGBoost ensemble artifacts for Attempt 7.

This exports the aggregated-jitter ensemble winner discovered in attempt7/study01
after refitting it on the full aggregated feature table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export attempt7 aggregated-jitter ensemble models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML file did not contain an object: {}".format(path))
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fit_rf(params: Dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params.get("max_depth") is None else int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params.get("max_features", "sqrt"),
        random_state=int(params["random_state"]),
        n_jobs=int(params.get("n_jobs", 1)),
    )


def _fit_xgb(params: Dict[str, Any]) -> Any:
    if XGBRegressor is None:
        raise RuntimeError("xgboost could not be imported: {}".format(XGB_IMPORT_ERROR))
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        random_state=int(params["random_state"]),
        n_jobs=int(params.get("n_jobs", 1)),
        tree_method=str(params.get("tree_method", "hist")),
        verbosity=int(params.get("verbosity", 0)),
    )


def _load_feature_rows(path: Path, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError("Feature table is empty: {}".format(path))
    numeric_feature_names = [name for name in feature_names if "=" not in name]
    one_hot_feature_names = [name for name in feature_names if "=" in name]
    parsed: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["true_distance_m"] = float(row["true_distance_m"])
        for name in numeric_feature_names:
            row[name] = float(row[name])
        for name in one_hot_feature_names:
            base_name, level = name.split("=", 1)
            row[name] = 1.0 if str(row.get(base_name, "")) == level else 0.0
        parsed.append(row)
    return parsed


def _matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def _target_values(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["true_distance_m"]) for row in rows], dtype=np.float64)


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
    config = _read_yaml(config_path)
    sources = dict(config["sources"])

    study01_summary = _read_json(_resolve_repo_path(str(sources["attempt7_summary_json"])))
    feature_names = list(study01_summary["selected_feature_names"])
    rows = _load_feature_rows(
        _resolve_repo_path(str(sources["feature_table_csv"])),
        feature_names,
    )

    rf_candidate_name = str(study01_summary["rf_winner_candidate_name"])
    xgb_candidate_name = str(study01_summary["xgb_winner_candidate_name"])
    rf_weight = float(dict(study01_summary["best_blend_search_row"])["rf_weight"])
    xgb_weight = float(dict(study01_summary["best_blend_search_row"])["xgb_weight"])

    rf_params_by_name = {
        "rf_shallow": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "random_state": 20260520, "n_jobs": 1},
        "rf_medium": {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 3, "max_features": "sqrt", "random_state": 20260520, "n_jobs": 1},
        "rf_deep": {"n_estimators": 300, "max_depth": 24, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 20260520, "n_jobs": 1},
        "rf_unbounded": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 20260520, "n_jobs": 1},
    }
    xgb_params_by_name = {
        "xgb_shallow": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 4.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "random_state": 20260520, "n_jobs": 1, "tree_method": "hist", "verbosity": 0},
        "xgb_medium": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 2.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "random_state": 20260520, "n_jobs": 1, "tree_method": "hist", "verbosity": 0},
        "xgb_deep": {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 2.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "random_state": 20260520, "n_jobs": 1, "tree_method": "hist", "verbosity": 0},
        "xgb_regularized": {"n_estimators": 450, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 3.0, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0, "reg_alpha": 0.1, "random_state": 20260520, "n_jobs": 1, "tree_method": "hist", "verbosity": 0},
    }
    rf_params = dict(rf_params_by_name[rf_candidate_name])
    xgb_params = dict(xgb_params_by_name[xgb_candidate_name])

    output_root = _resolve_repo_path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    model_dir = output_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    x = _matrix(rows, feature_names)
    y = _target_values(rows)
    rf_model = _fit_rf(rf_params)
    rf_model.fit(x, y)
    xgb_model = _fit_xgb(xgb_params)
    xgb_model.fit(x, y)

    schema = {
        "continuous_feature_names": list(feature_names),
        "categorical_feature_names": [],
        "categorical_levels": {},
        "tabular_feature_names": list(feature_names),
        "rf_candidate_name": rf_candidate_name,
        "xgb_candidate_name": xgb_candidate_name,
        "rf_params": dict(rf_params),
        "xgb_params": dict(xgb_params),
        "feature_representation": "aggregated_jitter_summary",
        "selected_source_model_name": str(study01_summary["selected_source_model_name"]),
    }
    _write_json(reports_dir / "model_schema.json", schema)

    exported_scopes: Dict[str, Any] = {}
    for scope_name in list(dict(config.get("export", {})).get("scopes", [])):
        paths = _dump_scope_artifacts(
            scope_name=scope_name,
            model_dir=model_dir,
            rf_model=rf_model,
            xgb_model=xgb_model,
            schema=schema,
            blend_weights={"rf_weight": rf_weight, "xgb_weight": xgb_weight},
            train_row_count=len(rows),
        )
        exported_scopes[scope_name] = {"train_rows": len(rows), "paths": paths}

    summary = {
        "study_name": str(config["study_name"]),
        "feature_table_csv": str(_resolve_repo_path(str(sources["feature_table_csv"]))),
        "num_rows": len(rows),
        "num_features": len(feature_names),
        "rf_candidate_name": rf_candidate_name,
        "xgb_candidate_name": xgb_candidate_name,
        "blend_weights": {"rf_weight": rf_weight, "xgb_weight": xgb_weight},
        "model_schema_json": str(reports_dir / "model_schema.json"),
        "exported_scopes": exported_scopes,
    }
    _write_json(reports_dir / "summary.json", summary)
    print("[attempt7-study02] Wrote summary: {}".format(reports_dir / "summary.json"))
    return summary


def main() -> None:
    args = parse_args()
    export_models(config_path=args.config)


if __name__ == "__main__":
    main()
