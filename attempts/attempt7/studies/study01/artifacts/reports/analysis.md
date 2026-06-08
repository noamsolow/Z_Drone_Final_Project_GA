# Attempt 7 Study 01

This study reuses the exact aggregated-jitter feature family from the Attempt 3 Study 03 `raw_depth_geometry_metadata_rf` winner,
but remaps the Attempt 4 source-image split onto the one-row-per-image aggregated table so the RF/XGB/blend comparison stays as fair as possible.

## Setup

- Feature source model: `raw_depth_geometry_metadata_rf`
- Number of selected features: `21`
- Dev rows: `12804`
- Test rows: `2260`
- RF winner candidate: `rf_deep`
- XGB winner candidate: `xgb_deep`

## Main Result

- Attempt 7 winner: `tuned_weight_blend`
- Attempt 7 winner test MAE: `5.9725`
- Attempt 7 winner test mean relative error: `0.0778`

## Comparison To Attempt 4

- Attempt 4 tuned blend test MAE: `7.6347`
- Attempt 4 equal-weight blend test MAE: `7.6325`
- Attempt 7 delta vs Attempt 4 tuned blend: `-1.6622`
- Attempt 7 delta vs Attempt 4 equal blend: `-1.6600`