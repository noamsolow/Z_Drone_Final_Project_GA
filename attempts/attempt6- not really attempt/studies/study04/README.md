# Attempt6 Study04: Milestone Progress Graphs

This study collects the major milestone models from earlier attempts and
generates the first two graph families discussed in the report-planning step:

- overall project progress by model stage
- synthetic milestone performance by true distance

## Graph Families

### 1. Overall Progress

These plots show the major milestone models in project order:

- `scale_only`
- `depth_only_linear`
- `depth_geometry_metadata_linear`
- `stacked_rf_depth_geometry_metadata_linear`
- `raw_depth_geometry_metadata_rf`
- `raw_depth_geometry_metadata_rf` with noisy-bbox aggregation
- `equal_weight_blend`
- calibrated ensemble (`per_drone_quadratic`, honest 20% split evaluation)

Outputs:

- `artifacts/plots/overall_progress_mae.png`
- `artifacts/plots/overall_progress_mean_relative_error.png`

### 2. Synthetic Distance Curves

These plots show the synthetic milestone models by true distance in meters.
Only synthetic models are included here, so the real calibrated Nenrus model is
left out on purpose.

Outputs:

- `artifacts/plots/synthetic_distance_mae.png`
- `artifacts/plots/synthetic_distance_mean_relative_error.png`

## Reports

- `artifacts/reports/milestone_overview.csv`
- `artifacts/reports/synthetic_distance_curves.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/analysis.md`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study04/run_study_04.py"
```
