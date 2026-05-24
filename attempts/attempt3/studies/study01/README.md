# Study 01: Linear Stacked Depth Comparison

This study is the first step of `attempt3`.

Its goal is simple:

- keep the old `attempt1` linear calibration setup
- keep the same geometry and metadata features
- replace the old raw depth signal with the learned output of the small Study 06 random forest
- measure whether that replacement improves metric-distance regression

The comparison is intentionally apples-to-apples:

- same linear-model family as the old baseline
- same bbox geometry inputs
- same `weather` / `time_of_day` metadata
- only the depth signal changes

## Source Inputs

The study reuses cached CSVs instead of recomputing depth maps:

- [attempt1 features.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt1/artifacts/features/features.csv)
- [study06 depth_only_feature_table.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/fused_features/depth_only_feature_table.csv)
- [study06 subset_random_forest_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_metrics.csv)

## What The Code Does

1. Load the old `attempt1` per-image table with:
   - `relative_depth`
   - bbox geometry
   - `weather`
   - `time_of_day`
2. Load the Study 06 fused depth-only feature table.
3. Merge both tables by `image_path`.
4. Read the selected lower-model feature set from Study 06 model metrics.
5. Refit the small random forest inside each outer split using only cached Study 06 depth features.
6. Use its split-safe predictions as a new feature named `depth_lower_model_prediction`.
7. Compare linear models such as:
   - `raw_depth_geometry_metadata_linear`
   - `stacked_rf_depth_geometry_metadata_linear`

## Leakage Handling

The stacked feature is not taken from an in-sample RF prediction.

Instead, the code uses nested fitting:

- inside each outer fold, the small RF produces out-of-fold predictions for the outer-train rows
- the same small RF is then refit on the full outer-train rows and predicts the outer-validation rows
- the stage-two linear model sees only those split-safe predictions

That keeps the comparison honest while still reusing all cached feature CSVs.

## Main Outputs

When the study is run, it writes:

- `artifacts/features/merged_feature_table.csv`
- `artifacts/features/summary.json`
- `artifacts/splits/test_split.json`
- `artifacts/splits/cv_folds.json`
- `artifacts/reports/cv_metrics.json`
- `artifacts/reports/test_metrics.json`
- `artifacts/reports/model_coefficients.csv`
- `artifacts/reports/grouped_error_summary.csv`
- `artifacts/reports/feature_correlations.csv`
- `artifacts/reports/cv_predictions.csv`
- `artifacts/reports/test_predictions.csv`
- `artifacts/reports/lower_model_feature_importances.csv`
- `artifacts/reports/lower_model_summary.json`
- `artifacts/reports/summary.json`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt3/studies/study01/run_study_01.py"
```

Optional config override:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt3/studies/study01/run_study_01.py" --config "attempts/attempt3/studies/study01/config.yaml"
```
