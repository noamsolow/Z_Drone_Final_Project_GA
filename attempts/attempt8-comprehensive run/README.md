# Attempt 8: Comprehensive Unified Synthetic Rerun

This folder contains the unified rerun for the poster's synthetic model family.

The goal is to rerun the main synthetic models under one shared benchmark and
one shared split, while preserving each model's method definition as closely as
possible.

## What This Rerun Includes

Poster models:

- `scale only`
- `depth linear`
- `linear+bb`
- `improved linear`
- `RF`
- `RF+jitter`
- `ensemble`

Internal supporting model:

- `XGBoost+jitter`

## Frozen Method Choices

The rerun keeps these decisions fixed:

- `scale only` stays the original through-origin scale model on `relative_depth`
- `depth linear` stays the original linear model on `relative_depth`
- `linear+bb` stays the original linear model on `relative_depth + bbox geometry + metadata`
- `improved linear` stays the stacked linear model that uses the `attempt2/study06` lower RF prediction
- `RF` is rerun as `raw_depth_geometry_metadata_rf` with raw depth fixed to `bbox_only__inner50_median__object_depth`
- `RF+jitter` stays the aggregated jitter-summary RF family
- `ensemble` stays the aggregated jitter-summary RF + XGBoost blend

## Shared Benchmark Logic

The benchmark is the common source-image intersection across:

- the merged exact-feature table from `attempt1 + attempt2/study06`
- the aggregated jitter-summary feature table from `attempt3/study03`
- the `attempt4` source-image split universe

The split itself is inherited from `attempt4`, then filtered to the common
intersection.

This keeps all reported synthetic models on one unified image universe.

## Outputs

When the run finishes, artifacts are written under:

- `attempts/attempt8-comprehensive run/artifacts/`

Key outputs:

- `audit/common_manifest.csv`
- `audit/dropped_rows_audit.csv`
- `audit/coverage_summary.json`
- `features/exact_common_feature_table.csv`
- `features/aggregated_common_feature_table.csv`
- `splits/unified_split.json`
- `splits/unified_cv_folds.json`
- `reports/model_catalog.csv`
- `reports/model_metrics.csv`
- `reports/distance_range_metrics.csv`
- `reports/predictions.csv`
- `reports/grouped_error_summary.csv`
- `reports/true_distance_summary.csv`
- `reports/poster_test_ranking.csv`
- `reports/summary.json`

## Runtime Notes

This script does not recompute depth maps.

It does rerun:

- model fitting
- cross-validation
- held-out evaluation
- RF candidate sweep
- XGBoost candidate sweep
- blend search

Console progress is printed during:

- source loading
- benchmark alignment
- each outer CV fold
- lower-model cross-fit
- RF candidate evaluation
- XGBoost candidate evaluation
- blend selection
- held-out test fitting

## How To Run

Main one-command wrapper:

```powershell
.\attempts\attempt8-comprehensive run\run_attempt8.ps1
```

If your virtual environment is elsewhere:

```powershell
.\attempts\attempt8-comprehensive run\run_attempt8.ps1 -PythonExe "C:\path\to\python.exe"
```

Direct Python invocation:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt8-comprehensive run/run_attempt8.py" --config "attempts/attempt8-comprehensive run/config.yaml"
```

## Prerequisites

Expected Python packages:

- `numpy`
- `pyyaml`
- `scikit-learn`
- `xgboost`

If `xgboost` is missing, the aggregated ensemble part will fail.
