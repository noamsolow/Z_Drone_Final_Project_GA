# Depth-Weighted External Variants Analysis

## Question

Can we reduce the Nenrus geometry-domain failure by making relative depth more influential?

## Ensemble Results

- `depth_plus_normalized_geometry`: MAE `21.611m`, mean relative error `0.858`, mean signed error `21.611m`, within 10m `0.217`
- `baseline_saved_ensemble`: MAE `26.108m`, mean relative error `1.000`, mean signed error `26.108m`, within 10m `0.076`
- `depth_repeated_16_with_geometry`: MAE `26.426m`, mean relative error `1.009`, mean signed error `26.426m`, within 10m `0.061`
- `depth_repeated_8_with_geometry`: MAE `26.735m`, mean relative error `1.017`, mean signed error `26.735m`, within 10m `0.051`
- `depth_plus_no_size_geometry`: MAE `42.787m`, mean relative error `2.449`, mean signed error `42.584m`, within 10m `0.037`
- `depth_only`: MAE `43.225m`, mean relative error `2.589`, mean signed error `43.025m`, within 10m `0.041`

## Interpretation

The best result came from `depth_plus_normalized_geometry`, not from `depth_only` and not from simply duplicating the depth feature.
That means relative depth helps only when it is paired with some geometry, and the more important change was removing pixel-size geometry features that are especially domain-sensitive.

`depth_only` performed much worse than the saved baseline, so relative depth by itself is not strong enough to solve the external-domain problem.
Repeating the depth feature 8x or 16x also did not improve over the saved baseline, which suggests that simply forcing more feature-sampling attention onto depth is not the right fix.

The best variant still overpredicted every Nenrus sample, so the core issue remains a systematic domain calibration problem.
Depth weighting can reduce the error somewhat, but it does not replace real-domain calibration.
