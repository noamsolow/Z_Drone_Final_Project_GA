# Mini Study 05: Relative Depth Tables

## Goal

This mini study does not train a model.

It only summarizes the raw relative-depth values that Depth Anything produced
for the drone region.

The output is three tables:

- `middle` = `bbox_midpoint`
- `mean` = `bbox_mean`
- `median` = `inner50_median`

Each table uses:

- rows = exact ground-truth distances
- columns = zoom contexts
- cell value = average `object_depth`

## Data Source

By default, the script reads:

- `../study_04_depth_only_models_30_per_stratum/artifacts/features/representation_records.csv`

That file already contains all Study 04 zooms:

- `1x`
- `1.5x`
- `2x`
- `4x`
- `6x`
- `8x`
- `10x`
- `16x`
- `full`

## Outputs

The script writes:

- `artifacts/reports/relative_depth_tables.md`
- `artifacts/reports/relative_depth_table_middle.csv`
- `artifacts/reports/relative_depth_table_mean.csv`
- `artifacts/reports/relative_depth_table_median.csv`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/second attempt/studies/study_05_relative_depth_tables/run_mini_study_05.py"
```
