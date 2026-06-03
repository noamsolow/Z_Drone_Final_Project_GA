# Study 01: Export RF+XGBoost Ensemble Weights

This study turns the best `attempt4` model family into reusable model artifacts.

It does not use the new Nenrus dataset.

## Goal

The goal is to save fitted model weights/artifacts for:

- Random Forest winner from `attempt4/study01`
- XGBoost winner from `attempt4/study02`
- RF+XGBoost ensemble metadata and blend weights

The exported model family is:

- `raw_depth_geometry_metadata_rf`
- `raw_depth_geometry_metadata_xgb`
- ensemble over both predictions

## Export Scopes

The script exports two scopes:

- `dev_holdout`: trained only on Attempt 4 dev rows, then evaluated on the held-out Attempt 4 test rows.
- `full_dataset`: trained on all Attempt 4 rows, intended for future external inference.

The `dev_holdout` artifact is useful for sanity-checking that the saved model
reproduces the known Attempt 4 result.

The `full_dataset` artifact is the one we should use for the next external
dataset evaluation, because it uses all original training data and still does
not train on the external dataset.

## Outputs

Expected outputs:

- `artifacts/models/dev_holdout/rf_model.joblib`
- `artifacts/models/dev_holdout/xgb_model.joblib`
- `artifacts/models/dev_holdout/ensemble.joblib`
- `artifacts/models/full_dataset/rf_model.joblib`
- `artifacts/models/full_dataset/xgb_model.joblib`
- `artifacts/models/full_dataset/ensemble.joblib`
- `artifacts/reports/dev_holdout_test_predictions.csv`
- `artifacts/reports/dev_holdout_test_metrics.json`
- `artifacts/reports/model_schema.json`
- `artifacts/reports/summary.json`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt5/studies/study01/export_ensemble_models.py"
```

