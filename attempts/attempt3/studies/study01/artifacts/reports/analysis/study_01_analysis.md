# Study 01 Analysis

## Scope

- Shared rows used: `13864`
- Dev rows: `11784`
- Test rows: `2080`
- Selected lower model: `random_forest_top_24`
- Lower-model depth features: `15`

## Main Result

- Winner: `Stacked RF Depth + Geometry + Metadata`
- Test MAE: `12.55m`
- Test mean relative error: `19.68%`
- CV OOF MAE: `12.78m`
- CV OOF mean relative error: `20.49%`

## Improvement

- Versus `Raw Depth + Geometry + Metadata`: `+1.03m` MAE improvement and `+1.53%` relative-error improvement.
- Versus `Geometry + Metadata`: `+1.30m` MAE improvement and `+1.89%` relative-error improvement.
- Versus `Small RF Depth Only`: `+8.35m` MAE improvement and `+14.69%` relative-error improvement.

## Files

- `study_01_analysis.md`
- `exact_distance_test_metrics.csv`
- `overall_model_metrics.csv`
- `distance_range_test_metrics.csv`
- `stacked_improvement_by_distance.csv`
- `exact_distance_mae_all_models.png`
- `exact_distance_relative_error_all_models.png`
- `overall_model_comparison.png`
- `stacked_improvement_mae_by_distance.png`
- `stacked_improvement_relative_error_by_distance.png`
- `distance_range_comparison.png`
- `exact_distance_signed_error_all_models.png`
