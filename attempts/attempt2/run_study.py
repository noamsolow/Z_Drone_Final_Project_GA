"""
Study runner for the second-attempt depth-representation comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calibration.depth_representation_features import (
    DEPTH_SCORE_FIELD_NAMES,
    read_depth_representation_records_csv,
)
from pipeline.calibration.depth_representation_study import evaluate_depth_representation_methods


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


def run_study(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    resolved_config = dict(config or load_config(config_path))
    output_root = Path(resolved_config["output_root"])
    features_csv_path = output_root / "features" / "representation_records.csv"
    if not features_csv_path.exists():
        raise FileNotFoundError(
            "Expected cached representation records at {}. "
            "Run attempts/attempt2/run_extract_features.py first.".format(features_csv_path)
        )

    records = read_depth_representation_records_csv(features_csv_path)
    study_config = resolved_config.get("study", {})
    score_fields = list(study_config.get("score_fields", DEPTH_SCORE_FIELD_NAMES))
    progress_log_every_methods = max(1, int(study_config.get("progress_log_every_methods", 1)))

    if verbose:
        print(
            "[second-attempt] Starting study ranking\n"
            "  config: {}\n"
            "  output_root: {}\n"
            "  features_csv: {}\n"
            "  num_records: {}\n"
            "  score_fields: {}".format(
                config_path or DEFAULT_CONFIG_PATH,
                output_root,
                features_csv_path,
                len(records),
                ", ".join(score_fields),
            )
        )

    def _progress_callback(payload: Dict[str, Any]) -> None:
        if not verbose:
            return
        event = payload.get("event", "")
        if event != "method_complete":
            return
        method_index = int(payload["method_index"])
        total_methods = int(payload["total_methods"])
        if method_index == 1 or method_index == total_methods or method_index % progress_log_every_methods == 0:
            print(
                "[second-attempt] Ranked method {}/{}\n"
                "  method_id: {}\n"
                "  loocv_mae: {:.4f}\n"
                "  loocv_mean_relative_error: {:.4f}\n"
                "  loocv_r2: {:.4f}".format(
                    method_index,
                    total_methods,
                    payload["method_id"],
                    float(payload["loocv_mae"]),
                    float(payload["loocv_mean_relative_error"]),
                    float(payload["loocv_r2"]),
                )
            )

    study_result = evaluate_depth_representation_methods(
        records,
        score_fields=score_fields,
        progress_callback=_progress_callback,
    )

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(study_result["ranking"], reports_dir / "method_metrics.csv")
    _write_csv_rows(study_result["coefficient_rows"], reports_dir / "method_coefficients.csv")
    _write_csv_rows(study_result["prediction_rows"], reports_dir / "loo_predictions.csv")
    _write_csv_rows(study_result["grouped_summary_rows"], reports_dir / "grouped_error_summary.csv")
    _write_csv_rows(
        [row for row in study_result["grouped_summary_rows"] if row.get("grouping") == "distance_range"],
        reports_dir / "distance_range_metrics.csv",
    )

    ranking = study_result["ranking"]
    winner = study_result["winner"]
    runner_up = dict(ranking[1]) if len(ranking) > 1 else {}
    gap_to_runner_up = (
        {
            "loocv_mae": float(runner_up["loocv_mae"]) - float(winner["loocv_mae"]),
            "loocv_mean_relative_error": float(runner_up["loocv_mean_relative_error"])
            - float(winner["loocv_mean_relative_error"]),
            "loocv_rmse": float(runner_up["loocv_rmse"]) - float(winner["loocv_rmse"]),
        }
        if winner and runner_up
        else {}
    )

    study_summary = {
        "study_name": resolved_config.get("study_name", output_root.name),
        "winner": winner,
        "runner_up": runner_up,
        "gap_to_runner_up": gap_to_runner_up,
        "top_5_methods": ranking[:5],
        "num_methods": len(ranking),
        "score_fields": score_fields,
        "best_by_context": study_result["best_by_context"],
        "best_by_aggregation": study_result["best_by_aggregation"],
        "best_by_score_field": study_result["best_by_score_field"],
        "ranking_rule": {
            "primary": "loocv_mae",
            "tie_breakers": [
                "loocv_mean_relative_error",
                "spearman_correlation",
                "method_id",
            ],
        },
    }
    study_summary_path = reports_dir / "study_summary.json"
    study_summary_path.write_text(json.dumps(study_summary, indent=2), encoding="utf-8")

    if verbose and winner:
        print(
            "[second-attempt] Study complete\n"
            "  winner: {}\n"
            "  loocv_mae: {:.4f}\n"
            "  loocv_mean_relative_error: {:.4f}\n"
            "  loocv_r2: {:.4f}\n"
            "  gap_to_runner_up_mae: {:.4f}\n"
            "  reports_dir: {}".format(
                winner.get("method_id", ""),
                float(winner["loocv_mae"]),
                float(winner["loocv_mean_relative_error"]),
                float(winner["loocv_r2"]),
                float(gap_to_runner_up.get("loocv_mae", 0.0)),
                reports_dir,
            )
        )

    return {
        "study_summary_json": str(study_summary_path),
        "method_metrics_csv": str(reports_dir / "method_metrics.csv"),
        "winner_method_id": winner.get("method_id", ""),
        "num_methods": len(ranking),
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
    result = run_study(config_path=args.config, verbose=not args.quiet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
