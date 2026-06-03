# Study 02: Noisy-BBox Expanded Rows XGBoost

This study is the XGBoost counterpart to `attempt4/study01`.
By default it seeds its feature table from `study01` when that CSV already
exists, so we can compare learners without recomputing identical feature rows.

## Core Idea

It keeps the exact same noisy expanded-row setup:

- generate several jittered bboxes per image
- recompute geometry and depth for each jitter
- keep each jitter as its own row
- split all rows from the same source image together

So the experiment changes the learner family, not the data construction.

## Current Jitter Mix

The default Study 02 setup uses:

- original
- slightly smaller
- slightly larger
- more larger

This version is scale-only: no directional shifts.

## Feature Setup

Contexts:

- `bbox_only`
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- `bbox_expand_4x`

Aggregation methods inside each jittered bbox:

- `bbox_midpoint`
- `bbox_mean`
- `inner50_median`

For each jitter row, the study stores:

- one raw geometry family
- one raw depth feature family

There is no jitter aggregation step.

## Models Compared

- `raw_depth_only_xgb`
- `geometry_metadata_xgb`
- `raw_depth_geometry_metadata_xgb`
- `small_xgb_depth_only_xgb`
- `stacked_xgb_depth_geometry_metadata_xgb`

## Main Outputs

Feature artifacts:

- `artifacts/features/noisy_expanded_feature_table.csv`
- `artifacts/features/summary.json`
- `artifacts/features/selected_samples.csv`

Evaluation artifacts:

- `artifacts/reports/cv_metrics.json`
- `artifacts/reports/test_metrics.json`
- `artifacts/reports/cv_predictions.csv`
- `artifacts/reports/test_predictions.csv`
- `artifacts/reports/grouped_error_summary.csv`
- `artifacts/reports/upper_xgb_candidate_metrics.csv`
- `artifacts/reports/upper_xgb_selected_models.csv`
- `artifacts/reports/upper_xgb_feature_importances.csv`
- `artifacts/reports/lower_model_feature_importances.csv`
- `artifacts/reports/summary.json`

## Requirements

This study needs `xgboost` available in the project virtualenv.

Check:

```powershell
.\.venv\Scripts\python.exe -c "import xgboost; print(xgboost.__version__)"
```

## Run

Dry run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt4/studies/study02/run_study_02.py" --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt4/studies/study02/run_study_02.py"
```

If you change the jitter mix after a previous run, delete the existing
`artifacts/features/noisy_expanded_feature_table.csv` first. The runner also
rebuilds cached splits automatically when they no longer match the current row
set.
