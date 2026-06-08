# Attempt6 Study04 Analysis

## Overview Graphs

The overall progress graphs use the eight agreed milestone models.
They should be treated as a narrative project-progress figure rather than one strict apples-to-apples benchmark, because the final two stages switch to a harder detector-like setup and then to real calibrated Nenrus evaluation.

- Best overall milestone by MAE: `Attempt 5 Study04 per_drone_quadratic honest 20% split evaluation` at 3.0735 m
- Best synthetic milestone by MAE: `Attempt 3 Study03 raw_depth_geometry_metadata_rf` at 5.2105 m

## Synthetic Distance Curves

The synthetic distance-curve plots only use synthetic models and therefore support a cleaner distance-based comparison.
They are built directly from saved prediction rows and aggregate MAE and mean relative error at each true distance.

Included synthetic models:
- `Depth linear`
- `Improved linear`
- `Linear + BB`
- `Noisy-bbox RF`
- `RF`
- `RF+XGB ensemble`
- `Scale only`

The real calibrated ensemble is intentionally excluded from the synthetic distance curves.
