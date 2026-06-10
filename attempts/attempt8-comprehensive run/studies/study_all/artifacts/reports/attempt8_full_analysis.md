# Attempt 8 Full Analysis

## Was Attempt 8 rerun from start to finish?

Yes for the modeling pipeline, but not from raw images. Attempt 8 reran model fitting, cross-validation, held-out evaluation, the RF candidate sweep, the XGBoost candidate sweep, and the blend search. It did not recompute depth maps or rebuild the upstream feature tables from scratch.

In practice, that means Attempt 8 is a full rerun of the comparison benchmark from the feature-table stage through final metrics, not a full rerun of the entire computer-vision stack.

## Executive Summary

- `ensemble` is the best test model with `MAE=6.008` and `R2=0.949` on `2260` test rows.
- The gain over `XGBoost+jitter` is tiny: `0.015 m` MAE. The ensemble is best, but the XGBoost branch is carrying most of the signal.
- The gain over `RF+jitter` is real and meaningful: `0.379 m` MAE.
- The gain over exact-feature `RF` is larger: `1.180 m` MAE.
- The selected blend is `RF 17% + XGB 83%`.
- The unified benchmark contains `15064` shared rows, with `2260` test rows and `0` dropped rows.

## Overall Ranking

| Model | Role | Test MAE | Test R2 | Within 10m | P95 Abs Error |
| --- | --- | --- | --- | --- | --- |
| ensemble | poster | 6.008 | 0.949 | 81.0% | 18.43 |
| XGBoost+jitter | internal | 6.024 | 0.949 | 80.8% | 18.42 |
| RF+jitter | poster | 6.387 | 0.941 | 78.2% | 19.96 |
| RF | poster | 7.189 | 0.930 | 73.9% | 21.65 |
| improved linear | poster | 12.458 | 0.833 | 48.3% | 30.66 |
| linear+bb | poster | 13.202 | 0.810 | 45.9% | 33.85 |
| depth linear | poster | 32.285 | 0.001 | 16.6% | 72.52 |
| scale only | poster | 47.702 | -1.325 | 13.3% | 110.89 |

## CV to Test Stability

The strongest sign that Attempt 8 is trustworthy is how little the top models move from OOF CV to the holdout test split.

| Model | CV MAE | Test MAE | Gap |
| --- | --- | --- | --- |
| ensemble | 5.971 | 6.008 | 0.038 |
| XGBoost+jitter | 5.987 | 6.024 | 0.037 |
| RF+jitter | 6.370 | 6.387 | 0.017 |
| RF | 7.260 | 7.189 | -0.071 |
| improved linear | 12.562 | 12.458 | -0.104 |
| linear+bb | 13.263 | 13.202 | -0.062 |
| depth linear | 32.263 | 32.285 | 0.021 |
| scale only | 47.238 | 47.702 | 0.464 |

## Distance Range Breakdown

| Model | Near MAE | Mid MAE | Far MAE |
| --- | --- | --- | --- |
| ensemble | 3.37 | 6.96 | 9.18 |
| XGBoost+jitter | 3.42 | 6.95 | 9.17 |
| RF+jitter | 3.44 | 7.65 | 9.65 |
| RF | 3.81 | 8.55 | 11.05 |
| improved linear | 12.03 | 9.51 | 17.13 |
| linear+bb | 12.77 | 9.74 | 18.59 |
| depth linear | 36.70 | 11.44 | 52.87 |
| scale only | 35.16 | 41.49 | 77.18 |

The entire leaderboard gets worse as distance increases, but the trees degrade much more gracefully than the linear families.

## Model by Model

### ensemble

- Test: `MAE=6.008`, `R2=0.949`, `within10=81.0%`.
- Distance MAE: near `3.37`, mid `6.96`, far `9.18`.
- This is the overall winner, but only by a hair over XGBoost+jitter. Its biggest value is not a huge headline gain, but a stable best-in-run result with almost no CV-to-test drift.

### XGBoost+jitter

- Test: `MAE=6.024`, `R2=0.949`, `within10=80.8%`.
- Distance MAE: near `3.42`, mid `6.95`, far `9.17`.
- This is the best single model in Attempt 8. Its test MAE is nearly identical to the final ensemble, which means most of the blend's strength comes from this branch.

### RF+jitter

- Test: `MAE=6.387`, `R2=0.941`, `within10=78.2%`.
- Distance MAE: near `3.44`, mid `7.65`, far `9.65`.
- Aggregated jitter features give a clear gain over the exact-feature RF. The improvement is visible in all three bands: near 3.44, mid 7.65, far 9.65 MAE.

### RF

- Test: `MAE=7.189`, `R2=0.930`, `within10=73.9%`.
- Distance MAE: near `3.81`, mid `8.55`, far `11.05`.
- This is the first clearly competitive model. It handles near distances well, but its mid-range R2 is almost flat and far-distance underprediction is still noticeable.

### improved linear

- Test: `MAE=12.458`, `R2=0.833`, `within10=48.3%`.
- Distance MAE: near `12.03`, mid `9.51`, far `17.13`.
- The stacked lower-model feature helps consistently over linear+bb, especially in mid-range, but the linear family still cannot match the tree-based models.

### linear+bb

- Test: `MAE=13.202`, `R2=0.810`, `within10=45.9%`.
- Distance MAE: near `12.77`, mid `9.74`, far `18.59`.
- Adding bbox geometry and metadata creates the first major jump. It is much stronger than the pure depth baselines, but still biased high in near and mid and low in far.

### depth linear

- Test: `MAE=32.285`, `R2=0.001`, `within10=16.6%`.
- Distance MAE: near `36.70`, mid `11.44`, far `52.87`.
- Relative depth alone is not enough. The model sits near zero R2 and behaves only marginally better than the scale-only baseline.

### scale only

- Test: `MAE=47.702`, `R2=-1.325`, `within10=13.3%`.
- Distance MAE: near `35.16`, mid `41.49`, far `77.18`.
- This is the weakest baseline. It fails badly at medium and far distances, has strongly negative R2, and keeps a large underprediction bias overall.

## Failure Patterns

- The most persistent weak zone for the best models is the `far` regime, especially around `100m` to `150m`.
- For the ensemble specifically, the most painful cluster is `light_rain` in the `mid` bucket, where the MAE rises above `8m`.
- The largest outliers are concentrated around `80m` to `100m`, with several `light_rain` samples overshooting by more than `40m`.
- The linear families show a structural bias pattern: they tend to overpredict near and mid distances, then underpredict far distances.

## Methodological Caveats

- Attempt 8 did not recompute depth maps. It reused prior feature tables from Attempts 1, 2, 3, and 4.
- The benchmark now covers the full Attempt 8 universe: `15064` shared rows, `12804` dev rows, `2260` test rows, and `0` dropped rows.
- The original missing exact-feature rows were regenerated with the Attempt1 extraction path before this rerun, so the current model comparison is no longer biased by the previous `10AM`/short-distance dropout.

## Generated Plots

- `plots/attempt8_test_mae_ranking.svg`
- `plots/attempt8_test_r2_ranking.svg`
- `plots/attempt8_cv_vs_test_mae.svg`
- `plots/attempt8_distance_range_heatmap.svg`
- `plots/attempt8_condition_mae.svg`
- `plots/attempt8_top_models_exact_distance.svg`
