# Study 04 Analysis

## Headline

- Best single feature: `bbox_only__bbox_midpoint__object_depth` with `MAE 25.04m`
- Best linear multifeature model: `greedy_forward_ols_k8` with `MAE 24.10m`
- Best overall model: `random_forest_top_24` with `MAE 22.95m`
- Best tree beat best single by `-2.09m` MAE

## What Strengthened

- A learned depth-only lower model clearly beats the best single depth feature.
- Tree models are now the top group, not just one lucky row.
- The winning feature family is still centered on `bbox_only`, `1.5x`, and `2x`.

## What The Graphs Show

- `top_single_features.png`
  - The strongest standalone features are still tight or local-context views.
- `top_models.png`
  - Tree models dominate the top of the leaderboard.
- `best_single_linear_tree_summary.png`
  - Linear fusion helps, but the tree model helps more.
- `bootstrap_ci_mae.png`
  - The best tree stays below the best single and best linear across the 95% MAE intervals.
- `distance_range_comparison.png`
  - The main gains still matter most around the hard distance regimes.
- `weather_comparison.png` and `time_of_day_comparison.png`
  - Weather and time matter, but less than distance structure.
- `best_tree_feature_importances.png`
  - The final winner is still driven by local relative-depth features, not broad-scene ones.

## Paired Significance

- best_model_vs_best_single: mean MAE delta = -2.09, 95% CI [-2.67, -1.48]
- best_tree_vs_best_linear: mean MAE delta = -1.15, 95% CI [-1.63, -0.68]
