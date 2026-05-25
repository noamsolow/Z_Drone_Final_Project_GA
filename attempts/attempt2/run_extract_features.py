"""
Feature extraction entry point for the second-attempt depth-representation study.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.depth_representation_features import (
    build_depth_representation_dataset,
    summarize_depth_representation_records,
    write_depth_representation_records_csv,
    write_selected_sample_records_csv,
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
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    study_config = resolved_config.get("study", {})
    progress_log_every_samples = max(1, int(study_config.get("progress_log_every_samples", 1)))

    if verbose:
        print(
            "[second-attempt] Starting feature extraction\n"
            "  config: {}\n"
            "  dataset_root: {}\n"
            "  output_root: {}\n"
            "  sample_count_per_stratum: {}\n"
            "  sample_seed: {}".format(
                config_path or DEFAULT_CONFIG_PATH,
                resolved_config["dataset_root"],
                output_root,
                study_config.get("sample_count_per_stratum", 2),
                study_config.get("sample_seed", 0),
            )
        )

    if depth_model is None:
        depth_config = resolved_config.get("depth_model", {})
        if verbose:
            print(
                "[second-attempt] Loading depth model\n"
                "  model_name: {}\n"
                "  device: {}\n"
                "  local_files_only: {}".format(
                    depth_config.get("model_name"),
                    depth_config.get("device"),
                    depth_config.get("local_files_only", False),
                )
            )
        depth_model = load_depth_anything_v2(
            model_name=depth_config.get("model_name"),
            device=depth_config.get("device"),
            local_files_only=bool(depth_config.get("local_files_only", False)),
        )

    def _progress_callback(payload: Dict[str, Any]) -> None:
        if not verbose:
            return
        event = payload.get("event", "")
        if event == "selection_complete":
            print(
                "[second-attempt] Selected samples\n"
                "  available_samples: {}\n"
                "  selected_samples: {}\n"
                "  strata: {}\n"
                "  per_stratum: {}\n"
                "  sample_seed: {}".format(
                    payload["num_available_samples"],
                    payload["num_selected_samples"],
                    payload["num_strata"],
                    payload["sample_count_per_stratum"],
                    payload["sample_seed"],
                )
            )
            return
        if event == "sample_start":
            sample_index = int(payload["sample_index"])
            total_samples = int(payload["total_samples"])
            if sample_index == 1 or sample_index == total_samples or sample_index % progress_log_every_samples == 0:
                print(
                    "[second-attempt] Sample {}/{}\n"
                    "  distance: {} m\n"
                    "  weather: {}\n"
                    "  time_of_day: {}\n"
                    "  stratum: {}\n"
                    "  image: {}".format(
                        sample_index,
                        total_samples,
                        payload["true_distance_m"],
                        payload["weather"],
                        payload["time_of_day"],
                        payload["stratum_key"],
                        payload["image_path"],
                    )
                )
            return
        if event == "sample_complete":
            sample_index = int(payload["sample_index"])
            total_samples = int(payload["total_samples"])
            if sample_index == 1 or sample_index == total_samples or sample_index % progress_log_every_samples == 0:
                print(
                    "[second-attempt] Finished sample {}/{} | records so far: {}".format(
                        sample_index,
                        total_samples,
                        payload["records_written_so_far"],
                    )
                )

    records, selected_samples = build_depth_representation_dataset(
        dataset_root=resolved_config["dataset_root"],
        depth_model=depth_model,
        strict=bool(resolved_config.get("strict", True)),
        sample_count_per_stratum=int(study_config.get("sample_count_per_stratum", 2)),
        sample_seed=int(study_config.get("sample_seed", 0)),
        context_window_specs=study_config.get("context_windows"),
        aggregation_methods=study_config.get("aggregation_methods"),
        inner_bbox_scale=float(study_config.get("inner_bbox_scale", 0.5)),
        surrounding_bbox_scale=float(study_config.get("surrounding_bbox_scale", 1.8)),
        progress_callback=_progress_callback,
    )

    features_csv_path = write_depth_representation_records_csv(
        records,
        features_dir / "representation_records.csv",
    )
    selected_samples_csv_path = write_selected_sample_records_csv(
        selected_samples,
        features_dir / "selected_samples.csv",
    )

    summary = summarize_depth_representation_records(records, selected_samples)
    summary.update(
        {
            "study_name": resolved_config.get("study_name", output_root.name),
            "dataset_root": str(resolved_config["dataset_root"]),
            "sample_count_per_stratum_requested": int(study_config.get("sample_count_per_stratum", 2)),
            "sample_seed": int(study_config.get("sample_seed", 0)),
            "representation_records_csv": str(features_csv_path),
            "selected_samples_csv": str(selected_samples_csv_path),
        }
    )
    summary_path = features_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(
            "[second-attempt] Feature extraction complete\n"
            "  selected_samples_csv: {}\n"
            "  representation_records_csv: {}\n"
            "  summary_json: {}\n"
            "  num_selected_samples: {}\n"
            "  num_representation_rows: {}".format(
                selected_samples_csv_path,
                features_csv_path,
                summary_path,
                len(selected_samples),
                len(records),
            )
        )

    return {
        "representation_records_csv": str(features_csv_path),
        "selected_samples_csv": str(selected_samples_csv_path),
        "summary_json": str(summary_path),
        "num_selected_samples": len(selected_samples),
        "num_representation_rows": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output and print only the final JSON result.",
    )
    args = parser.parse_args()
    result = run_feature_extraction(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
