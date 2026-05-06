# Study 03 Full Analysis

## 1. Main Result

- Best single-scale method: `bbox_only__bbox_midpoint__object_depth` with `MAE 25.46 m`, `RMSE 31.24 m`, `R^2 0.337`
- Runner-up single-scale method: `bbox_expand_1_5x__bbox_midpoint__object_minus_ring` with `MAE 26.27 m`
- Best multiscale fusion model: `greedy_forward_ols_k6` with `MAE 24.41 m`, `RMSE 30.11 m`, `R^2 0.385`
- Fusion improvement vs best single: `MAE -1.05 m`, `RMSE -1.14 m`, `mean relative error -0.024`

## 2. Zoom-Level Findings

Best single method by zoom context:
- `full_image`: `full_image__bbox_midpoint__object_minus_ring` with MAE `31.22 m`
- `bbox_expand_16x`: `bbox_expand_16x__bbox_midpoint__object_depth` with MAE `31.75 m`
- `bbox_expand_10x`: `bbox_expand_10x__bbox_midpoint__object_depth` with MAE `31.45 m`
- `bbox_expand_8x`: `bbox_expand_8x__bbox_midpoint__object_depth` with MAE `31.03 m`
- `bbox_expand_6x`: `bbox_expand_6x__bbox_midpoint__object_minus_ring` with MAE `30.92 m`
- `bbox_expand_4x`: `bbox_expand_4x__bbox_midpoint__object_minus_ring` with MAE `29.92 m`
- `bbox_expand_2x`: `bbox_expand_2x__bbox_midpoint__object_minus_ring` with MAE `27.92 m`
- `bbox_expand_1_5x`: `bbox_expand_1_5x__bbox_midpoint__object_minus_ring` with MAE `26.27 m`
- `bbox_only`: `bbox_only__bbox_midpoint__object_depth` with MAE `25.46 m`

The strongest single zoom remained `bbox_only`, while the strongest context-relative zoom remained `bbox_expand_1_5x`. The newly added wide contexts (`6x`, `8x`, `10x`, `16x`) all performed much worse than the tight local zooms.

## 3. What Each Score Field Did as Zoom Changed

- `object_depth` stayed strongest in the tightest crop, especially with `bbox_midpoint`.
- `object_minus_ring` was strongest in medium local zooms such as `1.5x` and `2x`, where there is enough background for a meaningful ring.
- `object_depth_percentile_5_95` helped some tight-crop variants, but did not beat the best raw-depth method.
- `object_minus_ring_normalized` was usually weaker than the raw ring-difference version.
- Very wide zooms caused all score fields to drift toward worse MAE because the drone signal became diluted by scene content.

## 4. Distance-Range Behavior

- `near`: best single `MAE 27.70 m` vs best fusion `MAE 25.29 m`
- `mid`: best single `MAE 14.27 m` vs best fusion `MAE 16.12 m`
- `far`: best single `MAE 36.66 m` vs best fusion `MAE 33.98 m`

Interpretation: the fusion model mainly helped the `near` and `far` regimes, while `mid` became slightly worse. That means multiscale fusion reduced some of the distance-compression problem, but it did not solve it completely.

## 5. Weather and Time-of-Day

- `clear_sky`: best single `MAE 26.16 m` vs best fusion `MAE 25.12 m`
- `light_rain`: best single `MAE 24.76 m` vs best fusion `MAE 23.70 m`
- `10AM`: best single `MAE 25.69 m` vs best fusion `MAE 24.64 m`
- `8PM`: best single `MAE 25.23 m` vs best fusion `MAE 24.18 m`

Interpretation: weather and time-of-day effects were smaller than the distance-range effects. The models changed only modestly across `clear_sky` vs `light_rain` and `10AM` vs `8PM`, while `near` and `far` remained the dominant challenge.

## 6. What the Winning Fusion Model Actually Used

- `bbox_only__bbox_midpoint__object_depth`
- `bbox_only__inner50_median__object_depth_percentile_5_95`
- `bbox_only__inner50_median__object_depth`
- `bbox_only__bbox_midpoint__object_depth_percentile_5_95`
- `bbox_expand_2x__inner50_median__object_minus_ring`
- `bbox_expand_2x__bbox_midpoint__object_minus_ring`

That is the key lesson from Study 3: the model improved by combining a strong `bbox_only` core with `2x` local context correction. It did **not** improve by leaning on `6x`, `8x`, `10x`, `16x`, or `full_image`.

## 7. Overall Conclusion

Study 3 provides evidence that a multiscale method can beat the best single zoom, but the useful multiscale combination is **local multiscale**, not broad scene-scale fusion. The strongest practical direction now is to build the next model around `bbox_only` depth features plus `2x` local contrast features.

## 8. Graph Files

- `best_mae_by_context.png`
- `score_field_zoom_trends.png`
- `context_score_heatmap.png`
- `exact_distance_zoom_trends.png`
- `best_context_winner_per_exact_distance.png`
- `single_vs_fusion_summary.png`
- `best_single_vs_fusion_exact_distance.png`
- `distance_range_comparison.png`
- `weather_comparison.png`
- `time_of_day_comparison.png`
- `weather_distance_mae_heatmaps.png`
- `time_distance_mae_heatmaps.png`