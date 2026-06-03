# Study 01: Noisy-BBox Expanded Rows Random Forest

This study starts `attempt4`, and it directly follows the ideas from `attempt3/study03`.

## Core Idea

Study 03 used bbox perturbations, but then aggregated them back into one robust
feature row per image.

Study 01 changes that.

Instead of:

- making several jittered bboxes
- recomputing geometry and depth on each one
- then aggregating them

it now does:

- make several jittered bboxes
- recompute geometry and depth on each one
- keep each jitter as its own row

So the study simulates a more realistic detector setting where the bbox itself
is slightly imperfect and each imperfect detection is treated as a real case.

## Current Jitter Mix

The default Study 01 setup now uses:

- original
- slightly smaller
- slightly larger
- more larger

This version is scale-only: no left/right/up/down shifts, just one smaller box
and two larger boxes around the original.

## Important Design Choice

All jitter rows from the same original image are kept together in the same
split group.

That means:

- train cannot see one jitter of an image while test sees another jitter of the same image

This avoids leakage.

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

- `raw_depth_only_rf`
- `geometry_metadata_rf`
- `raw_depth_geometry_metadata_rf`
- `small_rf_depth_only_rf`
- `stacked_rf_depth_geometry_metadata_rf`

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
- `artifacts/reports/upper_rf_candidate_metrics.csv`
- `artifacts/reports/upper_rf_selected_models.csv`
- `artifacts/reports/upper_rf_feature_importances.csv`
- `artifacts/reports/lower_model_feature_importances.csv`
- `artifacts/reports/summary.json`

## Run

Dry run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt4/studies/study01/run_study_01.py" --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt4/studies/study01/run_study_01.py"
```

If you change the jitter mix after a previous run, delete the existing
`artifacts/features/noisy_expanded_feature_table.csv` first. The runner now
fails fast when it detects stale rows from an older jitter scheme.
