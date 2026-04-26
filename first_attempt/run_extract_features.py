"""
Feature extraction entry point for the first calibration study.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.features import (
    build_regression_dataset,
    summarize_regression_records,
    write_regression_records_csv,
)
from pipeline.depth.depth_model import load_depth_anything_v2


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_feature_extraction(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    depth_model: Any = None,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    if depth_model is None:
        depth_config = resolved_config.get("depth_model", {})
        depth_model = load_depth_anything_v2(
            model_name=depth_config.get("model_name"),
            device=depth_config.get("device"),
        )

    records = build_regression_dataset(
        dataset_root=resolved_config["dataset_root"],
        depth_model=depth_model,
        strict=bool(resolved_config.get("strict", True)),
        max_samples=resolved_config.get("max_samples"),
    )

    features_csv_path = write_regression_records_csv(records, features_dir / "features.csv")
    summary = summarize_regression_records(records)
    summary.update(
        {
            "dataset_root": str(resolved_config["dataset_root"]),
            "features_csv": str(features_csv_path),
        }
    )
    summary_path = features_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "features_csv": str(features_csv_path),
        "summary_json": str(summary_path),
        "num_records": len(records),
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
    result = run_feature_extraction(config_path=args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
