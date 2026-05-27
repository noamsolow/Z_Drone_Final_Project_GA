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
.\.venv\Scripts\python.exe "attempts/attempt2/studies/study_05_relative_depth_tables/run_mini_study_05.py"
```

## BBox-only Depth Maps

If you want a few concrete visual examples, there is also a helper script that:

- selects `2` images from `30m`
- selects `2` images from `100m`
- selects `2` images from `150m`
- crops each image to the exact drone bounding box only
- runs Depth Anything on that bbox-only crop
- saves both the depth-map image and an RGB-vs-depth panel

Run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt2/studies/study_05_relative_depth_tables/make_bbox_only_depth_maps.py" --local-files-only
```

Outputs:

- `artifacts/bbox_only_depth_maps/*_depth_map.png`
- `artifacts/bbox_only_depth_maps/*_panel.png`
- `artifacts/bbox_only_depth_maps/bbox_only_depth_maps_contact_sheet.png`
- `artifacts/bbox_only_depth_maps/selected_bbox_only_depth_maps.csv`
