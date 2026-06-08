from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attempts.attempt4.studies.study01.run_study_01 import run_study as run_attempt4_study_01


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Attempt6 Study03 raw depth aggregation ablation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Study03 YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare generated configs and report the planned ablation without rerunning the RF.",
    )
    parser.add_argument(
        "--verbose-child-runs",
        action="store_true",
        help="Print verbose logs from the child Attempt4 Study01 reruns.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _write_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _read_json(input_path: Path) -> Dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def _copy_if_present(source_path: Path, destination_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError("Required source artifact not found: {}".format(source_path))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _prepare_variant_seed_artifacts(
    source_config: Dict[str, Any],
    source_paths: Dict[str, Path],
    variant_output_root: Path,
) -> None:
    features_dir = variant_output_root / "features"
    splits_dir = variant_output_root / "splits"
    features_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_present(source_paths["feature_table_csv"], features_dir / "noisy_expanded_feature_table.csv")
    _copy_if_present(source_paths["feature_summary_json"], features_dir / "summary.json")
    _copy_if_present(source_paths["selected_samples_csv"], features_dir / "selected_samples.csv")
    _copy_if_present(source_paths["test_split_json"], splits_dir / "test_split.json")
    _copy_if_present(source_paths["cv_folds_json"], splits_dir / "cv_folds.json")

    # Keep a frozen copy of the historical source config next to each rerun.
    _write_json(source_config, variant_output_root / "source_attempt4_config.json")


def _build_variant_config(
    source_attempt4_config: Dict[str, Any],
    variant: Dict[str, Any],
    model_names: Sequence[str],
    variant_output_root: Path,
) -> Dict[str, Any]:
    variant_config = deepcopy(source_attempt4_config)
    variant_config["study_name"] = "{}__{}".format(
        source_attempt4_config.get("study_name", "attempt4_study01"),
        str(variant["name"]),
    )
    variant_config["output_root"] = str(variant_output_root.relative_to(REPO_ROOT))
    variant_config.setdefault("study", {})
    variant_config["study"]["raw_depth_feature_name"] = str(variant["raw_depth_feature_name"])
    variant_config.setdefault("modeling", {})
    variant_config["modeling"]["model_names"] = list(model_names)
    return variant_config


def _load_variant_metrics(variant_output_root: Path, variant: Dict[str, Any]) -> Dict[str, Any]:
    reports_dir = variant_output_root / "reports"
    summary = _read_json(reports_dir / "summary.json")
    cv_metrics = _read_json(reports_dir / "cv_metrics.json")
    test_metrics = _read_json(reports_dir / "test_metrics.json")
    selected_models = list(cv_metrics.get("models", {}).values())
    if len(selected_models) != 1:
        raise ValueError("Expected exactly one model in the ablation rerun.")
    model_name = str(summary["winner_model_name"])
    test_model_metrics = dict(test_metrics["models"][model_name]["overall"])
    cv_model_metrics = dict(cv_metrics["models"][model_name]["oof_overall"])
    return {
        "variant_name": str(variant["name"]),
        "variant_label": str(variant["label"]),
        "raw_depth_feature_name": str(variant["raw_depth_feature_name"]),
        "selected_candidate_name": str(summary["winner_candidate_name"]),
        "cv_oof_mae": float(cv_model_metrics["mae"]),
        "cv_oof_rmse": float(cv_model_metrics["rmse"]),
        "cv_oof_r2": float(cv_model_metrics["r2"]),
        "test_mae": float(test_model_metrics["mae"]),
        "test_rmse": float(test_model_metrics["rmse"]),
        "test_r2": float(test_model_metrics["r2"]),
        "reports_dir": str(reports_dir),
        "summary": summary,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
    }


def _comparison_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = list(results)
    baseline = results[0]
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "variant_name": result["variant_name"],
                "variant_label": result["variant_label"],
                "raw_depth_feature_name": result["raw_depth_feature_name"],
                "selected_candidate_name": result["selected_candidate_name"],
                "cv_oof_mae": _round_float(result["cv_oof_mae"]),
                "test_mae": _round_float(result["test_mae"]),
                "cv_oof_delta_vs_baseline": _round_float(result["cv_oof_mae"] - baseline["cv_oof_mae"]),
                "test_mae_delta_vs_baseline": _round_float(result["test_mae"] - baseline["test_mae"]),
                "cv_oof_pct_delta_vs_baseline": _round_float(
                    100.0 * (result["cv_oof_mae"] - baseline["cv_oof_mae"]) / baseline["cv_oof_mae"]
                ),
                "test_pct_delta_vs_baseline": _round_float(
                    100.0 * (result["test_mae"] - baseline["test_mae"]) / baseline["test_mae"]
                ),
            }
        )
    return rows


def _candidate_comparison_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        model_metrics = result["cv_metrics"]["models"]["raw_depth_geometry_metadata_rf"]
        for candidate_row in model_metrics.get("candidate_metric_rows", []):
            rows.append(
                {
                    "variant_name": result["variant_name"],
                    "raw_depth_feature_name": result["raw_depth_feature_name"],
                    **candidate_row,
                }
            )
    return rows


def _per_distance_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    distance_keys = set()
    for result in results:
        distance_keys.update(result["test_metrics"]["models"]["raw_depth_geometry_metadata_rf"]["per_distance_range"].keys())
    for distance_key in sorted(distance_keys):
        baseline_mae = None
        for result in results:
            metric = result["test_metrics"]["models"]["raw_depth_geometry_metadata_rf"]["per_distance_range"].get(distance_key)
            if metric is None:
                continue
            mae_value = float(metric["mae"])
            if result["variant_name"] == results[0]["variant_name"]:
                baseline_mae = mae_value
            rows.append(
                {
                    "distance_range": distance_key,
                    "variant_name": result["variant_name"],
                    "raw_depth_feature_name": result["raw_depth_feature_name"],
                    "mae": _round_float(mae_value),
                    "mean_relative_error": _round_float(float(metric["mean_relative_error"])),
                    "rmse": _round_float(float(metric["rmse"])),
                    "r2": _round_float(float(metric["r2"])),
                    "mae_delta_vs_baseline": _round_float(mae_value - baseline_mae) if baseline_mae is not None else 0.0,
                }
            )
    return rows


def _build_analysis_text(results: Sequence[Dict[str, Any]]) -> str:
    baseline = results[0]
    comparison = results[1]
    cv_delta = comparison["cv_oof_mae"] - baseline["cv_oof_mae"]
    test_delta = comparison["test_mae"] - baseline["test_mae"]

    if test_delta < -0.1 and cv_delta < -0.1:
        recommendation = (
            "The `bbox_midpoint` swap improves both CV and held-out test MAE for this exact RF setup. "
            "It is the cleaner raw-depth choice for future direct RF comparisons, but the change is still targeted enough "
            "that only depth-sensitive follow-up studies need reruns."
        )
    elif test_delta > 0.1 and cv_delta > -0.1:
        recommendation = (
            "The historical `inner50_median` choice remains the safer option for this exact RF setup. "
            "A full rerun of later studies is not justified by this ablation."
        )
    else:
        recommendation = (
            "The difference is small or mixed across CV and held-out test. "
            "This is better documented as a methodological caveat than treated as a reason to rerun the full study chain."
        )

    winner = comparison if comparison["test_mae"] < baseline["test_mae"] else baseline
    loser = baseline if winner is comparison else comparison

    lines = [
        "# Attempt6 Study03 Analysis",
        "",
        "## Setup",
        "",
        "This ablation reused the cached Attempt4 Study01 noisy expanded feature table and copied the original grouped holdout/CV split files.",
        "Only the raw depth feature feeding `raw_depth_geometry_metadata_rf` was changed.",
        "",
        "## Results",
        "",
        "- Baseline `{}`: CV OOF MAE = {:.6f}, test MAE = {:.6f}".format(
            baseline["raw_depth_feature_name"],
            baseline["cv_oof_mae"],
            baseline["test_mae"],
        ),
        "- Comparison `{}`: CV OOF MAE = {:.6f}, test MAE = {:.6f}".format(
            comparison["raw_depth_feature_name"],
            comparison["cv_oof_mae"],
            comparison["test_mae"],
        ),
        "- CV delta (`bbox_midpoint - inner50_median`) = {:+.6f}".format(cv_delta),
        "- Test delta (`bbox_midpoint - inner50_median`) = {:+.6f}".format(test_delta),
        "",
        "## Conclusion",
        "",
        "For this exact historical RF rerun, `{}` finished better on held-out test MAE than `{}`.".format(
            winner["raw_depth_feature_name"],
            loser["raw_depth_feature_name"],
        ),
        recommendation,
    ]
    return "\n".join(lines) + "\n"


def run_study(config_path: Path, dry_run: bool, verbose_child_runs: bool) -> Dict[str, Any]:
    resolved_config = load_config(config_path)
    output_root = _resolve_repo_path(str(resolved_config["output_root"]))
    reports_dir = output_root / "reports"
    generated_configs_dir = output_root / "generated_configs"
    runs_dir = output_root / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = _resolve_repo_path(str(resolved_config["source_attempt4"]["config_path"]))
    source_attempt4_config = load_config(source_config_path)
    source_paths = {
        key: _resolve_repo_path(str(value))
        for key, value in dict(resolved_config["source_attempt4"]).items()
        if key != "config_path"
    }
    model_names = list(resolved_config.get("modeling", {}).get("model_names", ["raw_depth_geometry_metadata_rf"]))
    variants = list(resolved_config.get("variants", []))
    if len(variants) != 2:
        raise ValueError("Study03 expects exactly two variants.")

    planned_rows: List[Dict[str, Any]] = []
    for variant in variants:
        variant_output_root = runs_dir / str(variant["name"])
        _prepare_variant_seed_artifacts(
            source_config=source_attempt4_config,
            source_paths=source_paths,
            variant_output_root=variant_output_root,
        )
        variant_config = _build_variant_config(
            source_attempt4_config=source_attempt4_config,
            variant=variant,
            model_names=model_names,
            variant_output_root=variant_output_root,
        )
        generated_config_path = generated_configs_dir / "{}.yaml".format(str(variant["name"]))
        generated_config_path.write_text(
            yaml.safe_dump(variant_config, sort_keys=False),
            encoding="utf-8",
        )
        planned_rows.append(
            {
                "variant_name": str(variant["name"]),
                "raw_depth_feature_name": str(variant["raw_depth_feature_name"]),
                "generated_config_path": str(generated_config_path),
                "variant_output_root": str(variant_output_root),
            }
        )

    _write_csv_rows(planned_rows, reports_dir / "planned_runs.csv")

    if dry_run:
        summary = {
            "study_name": str(resolved_config.get("study_name", "study03")),
            "dry_run": True,
            "planned_runs": planned_rows,
            "source_config_path": str(source_config_path),
            "model_names": list(model_names),
        }
        _write_json(summary, reports_dir / "summary.json")
        return summary

    variant_results: List[Dict[str, Any]] = []
    for variant in variants:
        generated_config_path = generated_configs_dir / "{}.yaml".format(str(variant["name"]))
        run_attempt4_study_01(
            config_path=generated_config_path,
            dry_run=False,
            verbose=verbose_child_runs,
        )
        variant_output_root = runs_dir / str(variant["name"])
        variant_results.append(_load_variant_metrics(variant_output_root, variant))

    comparison_rows = _comparison_rows(variant_results)
    candidate_rows = _candidate_comparison_rows(variant_results)
    per_distance_rows = _per_distance_rows(variant_results)
    analysis_text = _build_analysis_text(variant_results)

    _write_csv_rows(comparison_rows, reports_dir / "comparison_metrics.csv")
    _write_csv_rows(candidate_rows, reports_dir / "candidate_oof_comparison.csv")
    _write_csv_rows(per_distance_rows, reports_dir / "per_distance_range_comparison.csv")
    (reports_dir / "analysis.md").write_text(analysis_text, encoding="utf-8")

    baseline = variant_results[0]
    comparison = variant_results[1]
    winner = min(variant_results, key=lambda row: float(row["test_mae"]))
    summary = {
        "study_name": str(resolved_config.get("study_name", "study03")),
        "source_attempt4_config_path": str(source_config_path),
        "model_names": list(model_names),
        "baseline_variant_name": baseline["variant_name"],
        "comparison_variant_name": comparison["variant_name"],
        "baseline_raw_depth_feature_name": baseline["raw_depth_feature_name"],
        "comparison_raw_depth_feature_name": comparison["raw_depth_feature_name"],
        "baseline_cv_oof_mae": _round_float(baseline["cv_oof_mae"]),
        "baseline_test_mae": _round_float(baseline["test_mae"]),
        "comparison_cv_oof_mae": _round_float(comparison["cv_oof_mae"]),
        "comparison_test_mae": _round_float(comparison["test_mae"]),
        "cv_oof_delta_midpoint_minus_inner50": _round_float(comparison["cv_oof_mae"] - baseline["cv_oof_mae"]),
        "test_mae_delta_midpoint_minus_inner50": _round_float(comparison["test_mae"] - baseline["test_mae"]),
        "winner_variant_name": winner["variant_name"],
        "winner_raw_depth_feature_name": winner["raw_depth_feature_name"],
        "winner_selected_candidate_name": winner["selected_candidate_name"],
        "analysis_md": str(reports_dir / "analysis.md"),
        "comparison_metrics_csv": str(reports_dir / "comparison_metrics.csv"),
    }
    _write_json(summary, reports_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_study(
        config_path=Path(args.config),
        dry_run=bool(args.dry_run),
        verbose_child_runs=bool(args.verbose_child_runs),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
