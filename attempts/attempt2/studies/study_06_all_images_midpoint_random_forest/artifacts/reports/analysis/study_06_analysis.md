# Study 06 Analysis

## Headline

- Best single feature: `bbox_only__bbox_midpoint__object_depth` with `CV MAE 24.21m`.
- Best random forest: `random_forest_top_24` with `CV MAE 20.92m`, `RMSE 26.83m`, `R^2 0.505`.
- Improvement vs best single: `MAE -3.29m`.
- Relative to Study 04 best model, Study 06 changes MAE by `-2.03m`.

## Setup

- `num_image_rows = 15064`
- `num_available_features = 15`
- Score fields kept: `object_depth`
- Context subset came from the Study 06 config.

## Single-feature story

- The strongest single feature remained the same core winner from Study 04: `bbox_only__bbox_midpoint__object_depth`.
- The next-best alternatives stayed local: `1.5x`, `bbox_only med50`, and `2x midpoint`.
- `full_image` remained weak, and `bbox_mean` remained the weakest aggregation family.

## Model story

- The random forest still beat the best single feature by a clear margin.
- The `top_24` name is a legacy label; in this study it effectively used all `15` available features.
- Adding the last few lower-ranked features still helped slightly: `top_24` beat `top_12` on both MAE and RMSE.

## Distance behavior

- Best exact-distance region: `90m` with `MAE 13.36m`.
- Worst exact-distance region: `150m` with `MAE 45.82m`.
- Near distances show overprediction: at `20m`, mean signed error is `25.10m`.
- Far distances show underprediction: at `150m`, mean signed error is `-45.82m`.
- The middle band around `80m` to `100m` is much easier than the nearest and farthest distances.
- The overall gain is real, but the model still shows strong distance compression.

## Condition behavior

- Best weather condition: `clear_sky` with `MAE 20.54m`.
- Worst weather condition: `light_rain` with `MAE 21.30m`.
- Best time-of-day condition: `10AM` with `MAE 20.12m`.
- Worst time-of-day condition: `8PM` with `MAE 22.50m`.
- Hardest combined cell: `light_rain + 8PM` with `MAE 24.04m`.

## Interpretation

- Study 06 supports the same local-depth story as Study 04, but under a much larger and simpler full-data setting.
- The model does not need percentile or ring features to remain useful; raw `object_depth` alone still carries a strong signal.
- The remaining weakness is not the lack of a depth signal, but the imperfect conversion from relative depth to metric distance at the extremes.

## Files Generated

- `top_single_features.png`
- `top_random_forest_models.png`
- `best_single_by_context.png`
- `context_aggregation_heatmap.png`
- `best_random_forest_feature_importances.png`
- `exact_distance_behavior.png`
- `weather_time_comparison.png`
- `weather_time_heatmap.png`
- `study_06_analysis.md`