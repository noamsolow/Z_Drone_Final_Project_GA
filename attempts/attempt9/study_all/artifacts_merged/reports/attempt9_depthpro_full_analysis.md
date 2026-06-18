# Attempt 9 Depth Pro Full Analysis

## Was Attempt 9 Depth Pro rerun from start to finish?

Yes for the modeling pipeline. This rerun uses the configured feature tables as inputs, then reruns model fitting, cross-validation, held-out evaluation, the RF candidate sweep, the XGBoost candidate sweep, and the blend search.

For Attempt 9 Depth Pro, the upstream depth features were rebuilt before this benchmark with Depth Pro, while reusable non-depth fields were carried forward from the previous datasets where safe.

## Executive Summary

- `ensemble` is the best test model with `MAE=6.354` and `R2=0.944` on `2234` test rows.
- The gain over `XGBoost+jitter` is tiny: `0.027 m` MAE. The ensemble is best, but the XGBoost branch is carrying most of the signal.
- The gain over `RF+jitter` is real and meaningful: `0.322 m` MAE.
- The gain over exact-feature `RF` is larger: `1.149 m` MAE.
- The selected blend is `RF 21% + XGB 79%`.
- The unified benchmark contains `14865` shared rows, with `2234` test rows and `199` dropped rows.

## Overall Ranking

| Model | Role | Test MAE | Test R2 | Within 10m | P95 Abs Error |
| --- | --- | --- | --- | --- | --- |
| ensemble | poster | 6.354 | 0.944 | 78.7% | 19.18 |
| XGBoost+jitter | internal | 6.381 | 0.944 | 78.8% | 19.01 |
| RF+jitter | poster | 6.676 | 0.938 | 77.0% | 20.66 |
| RF | poster | 7.502 | 0.926 | 71.8% | 21.69 |
| improved linear | poster | 11.911 | 0.841 | 51.2% | 30.88 |
| linear+bb | poster | 13.339 | 0.805 | 46.2% | 34.63 |
| depth linear | poster | 32.372 | -0.000 | 16.8% | 72.55 |
| scale only | poster | 77.545 | -4.104 | 0.0% | 149.93 |

## CV to Test Stability

The strongest sign that this run is trustworthy is how little the top models move from OOF CV to the holdout test split.

| Model | CV MAE | Test MAE | Gap |
| --- | --- | --- | --- |
| ensemble | 6.245 | 6.354 | 0.108 |
| XGBoost+jitter | 6.269 | 6.381 | 0.112 |
| RF+jitter | 6.599 | 6.676 | 0.077 |
| RF | 7.522 | 7.502 | -0.020 |
| improved linear | 12.093 | 11.911 | -0.182 |
| linear+bb | 13.417 | 13.339 | -0.078 |
| depth linear | 32.329 | 32.372 | 0.043 |
| scale only | 77.307 | 77.545 | 0.238 |

## Distance Range Breakdown

| Model | Near MAE | Mid MAE | Far MAE |
| --- | --- | --- | --- |
| ensemble | 3.65 | 7.38 | 9.46 |
| XGBoost+jitter | 3.69 | 7.39 | 9.47 |
| RF+jitter | 3.76 | 7.97 | 9.76 |
| RF | 4.18 | 8.79 | 11.27 |
| improved linear | 10.61 | 10.27 | 16.24 |
| linear+bb | 13.00 | 9.64 | 18.82 |
| depth linear | 37.14 | 11.24 | 52.64 |
| scale only | 40.19 | 84.45 | 130.01 |

The entire leaderboard gets worse as distance increases, but the trees degrade much more gracefully than the linear families.

## Model by Model

### ensemble

- Test: `MAE=6.354`, `R2=0.944`, `within10=78.7%`.
- Distance MAE: near `3.65`, mid `7.38`, far `9.46`.
- This is the overall winner, but only by a hair over XGBoost+jitter. Its biggest value is not a huge headline gain, but a stable best-in-run result with almost no CV-to-test drift.

### XGBoost+jitter

- Test: `MAE=6.381`, `R2=0.944`, `within10=78.8%`.
- Distance MAE: near `3.69`, mid `7.39`, far `9.47`.
- This is the best single model in this run. Its test MAE is nearly identical to the final ensemble, which means most of the blend's strength comes from this branch.

### RF+jitter

- Test: `MAE=6.676`, `R2=0.938`, `within10=77.0%`.
- Distance MAE: near `3.76`, mid `7.97`, far `9.76`.
- Aggregated jitter features give a clear gain over the exact-feature RF. The improvement is visible in all three bands: near 3.76, mid 7.97, far 9.76 MAE.

### RF

- Test: `MAE=7.502`, `R2=0.926`, `within10=71.8%`.
- Distance MAE: near `4.18`, mid `8.79`, far `11.27`.
- This is the first clearly competitive model. It handles near distances well, but its mid-range R2 is almost flat and far-distance underprediction is still noticeable.

### improved linear

- Test: `MAE=11.911`, `R2=0.841`, `within10=51.2%`.
- Distance MAE: near `10.61`, mid `10.27`, far `16.24`.
- The stacked lower-model feature helps consistently over linear+bb, especially in mid-range, but the linear family still cannot match the tree-based models.

### linear+bb

- Test: `MAE=13.339`, `R2=0.805`, `within10=46.2%`.
- Distance MAE: near `13.00`, mid `9.64`, far `18.82`.
- Adding bbox geometry and metadata creates the first major jump. It is much stronger than the pure depth baselines, but still biased high in near and mid and low in far.

### depth linear

- Test: `MAE=32.372`, `R2=-0.000`, `within10=16.8%`.
- Distance MAE: near `37.14`, mid `11.24`, far `52.64`.
- Relative depth alone is not enough. The model sits near zero R2 and behaves only marginally better than the scale-only baseline.

### scale only

- Test: `MAE=77.545`, `R2=-4.104`, `within10=0.0%`.
- Distance MAE: near `40.19`, mid `84.45`, far `130.01`.
- This is the weakest baseline. It fails badly at medium and far distances, has strongly negative R2, and keeps a large underprediction bias overall.

## Failure Patterns

- The most persistent weak zone for the best models is the `far` regime, especially around `100m` to `150m`.
- For the ensemble specifically, the most painful cluster is `light_rain` in the `mid` bucket, where the MAE rises above `8m`.
- The largest outliers are concentrated around `80m` to `100m`, with several `light_rain` samples overshooting by more than `40m`.
- The linear families show a structural bias pattern: they tend to overpredict near and mid distances, then underpredict far distances.

## Methodological Caveats

- This study_all rerun does not invoke Depth Pro directly; it consumes the already-built Attempt 9 Depth Pro feature tables.
- Non-depth fields were reused from the previous datasets where safe; the depth-derived features were recomputed with Depth Pro.
- The benchmark now covers the full Attempt 9 Depth Pro universe: `14865` shared rows, `12631` dev rows, `2234` test rows, and `199` dropped rows.

## Generated Plots

- `plots/attempt9_depthpro_test_mae_ranking.svg`
- `plots/attempt9_depthpro_test_r2_ranking.svg`
- `plots/attempt9_depthpro_cv_vs_test_mae.svg`
- `plots/attempt9_depthpro_distance_range_heatmap.svg`
- `plots/attempt9_depthpro_condition_mae.svg`
- `plots/attempt9_depthpro_top_models_exact_distance.svg`
