# Attempt 9 Depth Pro Full Analysis

## Was Attempt 9 Depth Pro rerun from start to finish?

Yes for the modeling pipeline. This rerun uses the configured feature tables as inputs, then reruns model fitting, cross-validation, held-out evaluation, the RF candidate sweep, the XGBoost candidate sweep, and the blend search.

For Attempt 9 Depth Pro, the upstream depth features were rebuilt before this benchmark with Depth Pro, while reusable non-depth fields were carried forward from the previous datasets where safe.

## Executive Summary

- `ensemble` is the best test model with `MAE=6.341` and `R2=0.944` on `2260` test rows.
- The gain over `XGBoost+jitter` is tiny: `0.015 m` MAE. The ensemble is best, but the XGBoost branch is carrying most of the signal.
- The gain over `RF+jitter` is real and meaningful: `0.373 m` MAE.
- The gain over exact-feature `RF` is larger: `1.202 m` MAE.
- The selected blend is `RF 19% + XGB 81%`.
- The unified benchmark contains `15064` shared rows, with `2260` test rows and `0` dropped rows.

## Overall Ranking

| Model | Role | Test MAE | Test R2 | Within 10m | P95 Abs Error |
| --- | --- | --- | --- | --- | --- |
| ensemble | poster | 6.341 | 0.944 | 79.0% | 18.99 |
| XGBoost+jitter | internal | 6.355 | 0.944 | 79.1% | 19.02 |
| RF+jitter | poster | 6.714 | 0.936 | 76.9% | 20.56 |
| RF | poster | 7.543 | 0.925 | 71.4% | 22.08 |
| improved linear | poster | 11.943 | 0.840 | 51.5% | 30.74 |
| linear+bb | poster | 13.353 | 0.803 | 46.2% | 34.80 |
| depth linear | poster | 32.299 | -0.000 | 16.6% | 72.75 |
| scale only | poster | 77.383 | -4.110 | 0.0% | 149.93 |

## CV to Test Stability

The strongest sign that this run is trustworthy is how little the top models move from OOF CV to the holdout test split.

| Model | CV MAE | Test MAE | Gap |
| --- | --- | --- | --- |
| ensemble | 6.276 | 6.341 | 0.065 |
| XGBoost+jitter | 6.295 | 6.355 | 0.061 |
| RF+jitter | 6.634 | 6.714 | 0.080 |
| RF | 7.548 | 7.543 | -0.005 |
| improved linear | 12.109 | 11.943 | -0.166 |
| linear+bb | 13.445 | 13.353 | -0.092 |
| depth linear | 32.245 | 32.299 | 0.054 |
| scale only | 77.106 | 77.383 | 0.277 |

## Distance Range Breakdown

| Model | Near MAE | Mid MAE | Far MAE |
| --- | --- | --- | --- |
| ensemble | 3.64 | 7.42 | 9.44 |
| XGBoost+jitter | 3.69 | 7.42 | 9.43 |
| RF+jitter | 3.75 | 8.09 | 9.85 |
| RF | 4.21 | 8.88 | 11.36 |
| improved linear | 10.64 | 10.30 | 16.34 |
| linear+bb | 12.97 | 9.59 | 19.05 |
| depth linear | 36.74 | 11.44 | 52.84 |
| scale only | 40.39 | 84.59 | 130.01 |

The entire leaderboard gets worse as distance increases, but the trees degrade much more gracefully than the linear families.

## Model by Model

### ensemble

- Test: `MAE=6.341`, `R2=0.944`, `within10=79.0%`.
- Distance MAE: near `3.64`, mid `7.42`, far `9.44`.
- This is the overall winner, but only by a hair over XGBoost+jitter. Its biggest value is not a huge headline gain, but a stable best-in-run result with almost no CV-to-test drift.

### XGBoost+jitter

- Test: `MAE=6.355`, `R2=0.944`, `within10=79.1%`.
- Distance MAE: near `3.69`, mid `7.42`, far `9.43`.
- This is the best single model in this run. Its test MAE is nearly identical to the final ensemble, which means most of the blend's strength comes from this branch.

### RF+jitter

- Test: `MAE=6.714`, `R2=0.936`, `within10=76.9%`.
- Distance MAE: near `3.75`, mid `8.09`, far `9.85`.
- Aggregated jitter features give a clear gain over the exact-feature RF. The improvement is visible in all three bands: near 3.75, mid 8.09, far 9.85 MAE.

### RF

- Test: `MAE=7.543`, `R2=0.925`, `within10=71.4%`.
- Distance MAE: near `4.21`, mid `8.88`, far `11.36`.
- This is the first clearly competitive model. It handles near distances well, but its mid-range R2 is almost flat and far-distance underprediction is still noticeable.

### improved linear

- Test: `MAE=11.943`, `R2=0.840`, `within10=51.5%`.
- Distance MAE: near `10.64`, mid `10.30`, far `16.34`.
- The stacked lower-model feature helps consistently over linear+bb, especially in mid-range, but the linear family still cannot match the tree-based models.

### linear+bb

- Test: `MAE=13.353`, `R2=0.803`, `within10=46.2%`.
- Distance MAE: near `12.97`, mid `9.59`, far `19.05`.
- Adding bbox geometry and metadata creates the first major jump. It is much stronger than the pure depth baselines, but still biased high in near and mid and low in far.

### depth linear

- Test: `MAE=32.299`, `R2=-0.000`, `within10=16.6%`.
- Distance MAE: near `36.74`, mid `11.44`, far `52.84`.
- Relative depth alone is not enough. The model sits near zero R2 and behaves only marginally better than the scale-only baseline.

### scale only

- Test: `MAE=77.383`, `R2=-4.110`, `within10=0.0%`.
- Distance MAE: near `40.39`, mid `84.59`, far `130.01`.
- This is the weakest baseline. It fails badly at medium and far distances, has strongly negative R2, and keeps a large underprediction bias overall.

## Failure Patterns

- The most persistent weak zone for the best models is the `far` regime, especially around `100m` to `150m`.
- For the ensemble specifically, the most painful cluster is `light_rain` in the `mid` bucket, where the MAE rises above `8m`.
- The largest outliers are concentrated around `80m` to `100m`, with several `light_rain` samples overshooting by more than `40m`.
- The linear families show a structural bias pattern: they tend to overpredict near and mid distances, then underpredict far distances.

## Methodological Caveats

- This study_all rerun does not invoke Depth Pro directly; it consumes the already-built Attempt 9 Depth Pro feature tables.
- Non-depth fields were reused from the previous datasets where safe; the depth-derived features were recomputed with Depth Pro.
- The benchmark now covers the full Attempt 9 Depth Pro universe: `15064` shared rows, `12804` dev rows, `2260` test rows, and `0` dropped rows.

## Generated Plots

- `plots/attempt9_depthpro_test_mae_ranking.svg`
- `plots/attempt9_depthpro_test_r2_ranking.svg`
- `plots/attempt9_depthpro_cv_vs_test_mae.svg`
- `plots/attempt9_depthpro_distance_range_heatmap.svg`
- `plots/attempt9_depthpro_condition_mae.svg`
- `plots/attempt9_depthpro_top_models_exact_distance.svg`
