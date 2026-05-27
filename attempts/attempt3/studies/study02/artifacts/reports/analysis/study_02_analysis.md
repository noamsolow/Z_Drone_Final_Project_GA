# Study 02 Analysis

## Headline

- Winner: `raw_depth_geometry_metadata_rf` with candidate `rf_unbounded`.
- Winner test MAE: `7.4554m`
- Winner mean relative error: `9.38%`

## Key Takeaways

- The best final model was `raw_depth_geometry_metadata_rf`, not the stacked version.
- The stacked model stayed very strong, but it was slightly worse than the raw-depth big RF.
- Geometry remained the dominant signal once the upper model became a Random Forest.
- The main performance jump from Study 01 to Study 02 came from upgrading the top-level model family to Random Forest.

## Important Numbers

- Winner test MAE: `7.4554m`
- Stacked test MAE: `7.5402m`
- Geometry-only test MAE: `7.6911m`
- Small-RF-depth-only test MAE: `21.2744m`
- Raw-depth-only test MAE: `30.7453m`

## Interpretation

- The small RF clearly improved the depth-only signal, but that advantage did not translate into the best final stacked RF.
- Once the upper model could already learn strong non-linear corrections from geometry and raw depth together, the extra lower-model depth feature was no longer the winner.
