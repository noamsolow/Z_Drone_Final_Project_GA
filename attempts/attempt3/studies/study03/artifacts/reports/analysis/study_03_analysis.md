# Study 03 Analysis

## Headline

- Study 03 winner: `raw_depth_geometry_metadata_rf` with test MAE `5.2105m` and mean relative error `7.43%`.
- Study 02 winner: `raw_depth_geometry_metadata_rf` with test MAE `7.4554m` and mean relative error `9.38%`.
- Study 03 still favored `raw_depth_geometry_metadata_rf`, not the stacked model.

## Key Takeaways

- Noisy bbox aggregation improved overall accuracy substantially versus Study 02, despite making the boxes more realistic.
- The stacked model remained very strong, but it still did not beat `raw_depth_geometry_metadata_rf`.
- Geometry remained the dominant signal even after replacing exact bbox measurements with jitter-median and jitter-std features.
- The new `__jitter_std` geometry features became meaningfully important, indicating that robustness/stability information helped the model.

## Important Numbers

- Study 03 winner test MAE: `5.2105m`
- Study 03 stacked test MAE: `5.3318m`
- Study 03 geometry-only test MAE: `5.4021m`
- Gap between Study 03 winner and stacked: `0.1213m`
- Study 02 stacked test MAE: `7.5402m`

## Caveat

- Study 02 and Study 03 are not a perfect apples-to-apples comparison because Study 02 used the smaller shared intersection while Study 03 used all `15,064` images.
- Even so, the comparison is still very informative about how the noisy aggregated feature design changed the overall behavior.