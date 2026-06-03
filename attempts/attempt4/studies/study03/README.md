# Study 03: RF + XGBoost Blend

This study tests whether the best `attempt4/study01` random-forest model and
the best `attempt4/study02` XGBoost model complement each other when blended.

## Core Idea

The base models are not retrained here.

Instead, the study:

- loads the winner OOF predictions from `study01`
- loads the winner OOF predictions from `study02`
- aligns them row-by-row using `augmented_row_id`
- tunes a convex blend weight on the shared dev OOF predictions
- applies that same weight once to the shared held-out test predictions

This keeps the blend leakage-safe while staying fast.

## Models Compared

- `rf_winner_baseline`
- `xgb_winner_baseline`
- `equal_weight_blend`
- `tuned_weight_blend`

## Main Outputs

- `artifacts/reports/cv_metrics.json`
- `artifacts/reports/test_metrics.json`
- `artifacts/reports/blend_search.csv`
- `artifacts/reports/cv_predictions.csv`
- `artifacts/reports/test_predictions.csv`
- `artifacts/reports/grouped_error_summary.csv`
- `artifacts/reports/summary.json`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt4/studies/study03/run_study_03.py"
```
