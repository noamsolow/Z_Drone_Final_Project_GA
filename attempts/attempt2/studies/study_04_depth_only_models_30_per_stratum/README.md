# Study 04: Depth-Only Lower Model

## Goal

Study 04 is the first larger depth-only modeling study.

The goal is not to build the final production model yet.
The goal is to answer a narrower question:

- what is the strongest combination of relative-depth features by itself?

This study is meant to produce the lower model that can later feed a larger
final model as one clean learned depth signal.

## Scope

This study intentionally stays depth-only.

It does not use:

- bbox geometry as predictive inputs
- weather as predictive inputs
- time of day as predictive inputs

Those fields are still kept for analysis and grouped summaries, but they are
not allowed to help the model predict distance directly.

## Sample Size

- `30` images per exact `distance x weather x time` stratum
- `48` exact strata
- target total: `1,440` images

## Relative-Depth Feature Space

Contexts:

- `bbox_only`
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- `bbox_expand_4x`
- `bbox_expand_6x`
- `bbox_expand_8x`
- `bbox_expand_10x`
- `bbox_expand_16x`
- `full_image`

Aggregations:

- `inner50_median`
- `bbox_mean`
- `bbox_midpoint`

Score fields:

- `object_depth`
- `object_depth_percentile_5_95`
- `object_minus_ring`
- `object_minus_ring_normalized`

Total candidate depth-only features:

- `9 x 3 x 4 = 108`

## Workflow

1. Build a new Study 04 selected-sample list with `30` images per stratum.
2. Reuse any cached representation rows already available from Study 02 and
   Study 03.
3. Compute only the missing rows for the new larger sample.
4. Rank the single features.
5. Build one fused depth-only table with one row per image.
6. Compare multifeature models:
   - linear baselines
   - ridge baselines
   - tree-based models when the needed libraries are installed
7. Write confidence intervals, paired comparisons, grouped summaries, and
   feature-importance outputs.

## Outputs

- `artifacts/features/`
  - selected sample list
  - representation cache
  - feature-cache summary
- `artifacts/fused_features/`
  - one-row-per-image depth-only feature table
- `artifacts/reports/`
  - single-feature ranking under the Study 04 CV protocol
  - multifeature model metrics
  - coefficients and feature importances
  - out-of-fold predictions
  - grouped summaries
  - bootstrap intervals
  - paired model-comparison tables
  - overall study summary JSON

## Results

Headline results from the completed run:

- Best single feature:
  - `bbox_only__bbox_midpoint__object_depth`
  - `CV MAE = 25.04m`
- Best linear multifeature model:
  - `greedy_forward_ols_k8`
  - `CV MAE = 24.10m`
- Best overall model:
  - `random_forest_top_24`
  - `CV MAE = 22.95m`
  - `CV RMSE = 28.69m`
  - `CV R2 = 0.441`

Main interpretation:

- the learned depth-only lower model clearly beat the best single depth feature
- linear multifeature fusion helped, but tree models helped more
- the useful feature family is still concentrated around:
  - `bbox_only`
  - `bbox_expand_1_5x`
  - `bbox_expand_2x`
- the winner is still a local multiscale solution, not a broad-scene solution

## Significance

Paired bootstrap comparisons support that the gain is real:

- best model vs best single:
  - mean MAE delta `= -2.09m`
  - `95% CI = [-2.67, -1.48]`
- best tree vs best linear:
  - mean MAE delta `= -1.15m`
  - `95% CI = [-1.63, -0.68]`

So the tree-model improvement did not just appear by chance inside one noisy run.

## Analysis Package

Generated analysis folder:

- `artifacts/reports/analysis/`

Main analysis files:

- `study_04_analysis.md`
- `top_single_features.png`
- `top_models.png`
- `best_single_linear_tree_summary.png`
- `bootstrap_ci_mae.png`
- `distance_range_comparison.png`
- `weather_comparison.png`
- `time_of_day_comparison.png`
- `exact_distance_mae_comparison.png`
- `best_tree_feature_importances.png`
- `best_single_by_context.png`
- `context_score_heatmap.png`
- `weather_distance_heatmap_best_tree.png`
- `time_distance_heatmap_best_tree.png`

To regenerate the graph package:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt2/studies/study_04_depth_only_models_30_per_stratum/analyze_study_04.py"
```

## Dependency Note

The study is designed so that the linear parts still work without extra
packages, but the tree-based models require:

- `scikit-learn`
- optionally `xgboost`

If they are missing, the reports will mark those models as skipped instead of
failing silently.

## Run Guide

Recommended order:

1. Build or extend the Study 04 feature cache:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt2/studies/study_04_depth_only_models_30_per_stratum/run_extend_feature_cache.py"
```

2. Run the classic per-method ranking for the same Study 04 cache:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt2/run_study.py" --config "attempts/attempt2/studies/study_04_depth_only_models_30_per_stratum/config.yaml"
```

3. Run the dedicated depth-only lower-model search:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt2/studies/study_04_depth_only_models_30_per_stratum/run_depth_only_model_study.py"
```

If you want the tree-based models too, install:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn
.\.venv\Scripts\python.exe -m pip install xgboost
```

Core outputs to inspect:

- `artifacts/features/summary.json`
- `artifacts/reports/single_feature_cv_metrics.csv`
- `artifacts/reports/depth_only_model_metrics.csv`
- `artifacts/reports/depth_only_bootstrap_intervals.csv`
- `artifacts/reports/depth_only_paired_model_comparisons.csv`
- `artifacts/reports/depth_only_study_summary.json`
- `artifacts/reports/analysis/study_04_analysis.md`
