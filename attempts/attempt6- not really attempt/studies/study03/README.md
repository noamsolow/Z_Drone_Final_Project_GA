# Attempt6 Study03: Raw Depth Aggregation Ablation for the Old RF

This study reruns the old Attempt4 Study01 `raw_depth_geometry_metadata_rf`
setup on the same cached feature table and the same grouped train/test and CV
splits.

The only intended change is the raw depth feature fed into the big RF:

- baseline: `bbox_only__inner50_median__object_depth`
- comparison: `bbox_only__bbox_midpoint__object_depth`

## Goal

Check whether the old RF would perform better if we swapped only the raw depth
aggregation from `inner50_median` to `bbox_midpoint`, while keeping the rest of
the experiment fixed.

## What Stays Fixed

- same synthetic dataset rows as Attempt4 Study01
- same cached noisy expanded feature table
- same grouped holdout split
- same grouped CV folds
- same RF candidate grid
- same geometry and metadata features

## Outputs

- `artifacts/reports/comparison_metrics.csv`
- `artifacts/reports/candidate_oof_comparison.csv`
- `artifacts/reports/per_distance_range_comparison.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/analysis.md`

Each individual rerun also writes its own full Attempt4-style artifacts under:

- `artifacts/runs/inner50_median/`
- `artifacts/runs/bbox_midpoint/`

## Run

Dry run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study03/run_study_03.py" --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study03/run_study_03.py"
```
