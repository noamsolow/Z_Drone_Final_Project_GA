# Study 02: External Nenrus Evaluation

This study evaluates the exported `attempt5/study01` RF+XGBoost ensemble on the
prepared Nenrus drone-distance dataset.

The external dataset is not used for training.

## Goal

The goal is to answer:

- does our exported model work on real-drone images from a different dataset?
- where does it fail?
- how does error change by drone type/source?
- how does error change by distance?

## Input Dataset

Expected prepared dataset:

```text
nenrus-Dataset-of-Drone-Distance-Estimation-a5aac8f/
  prepared_by_drone_type/
    Kongsberg/
    Vestfold/
    manifest.csv
```

## Model

The study loads:

```text
attempts/attempt5/studies/study01/artifacts/models/full_dataset/ensemble.joblib
```

That metadata points to:

- `rf_model.joblib`
- `xgb_model.joblib`
- blend weights
- feature schema

## Important Metadata Limitation

The original model was trained with metadata features:

- `weather`
- `time_of_day`

The Nenrus dataset does not provide those metadata values, so this study sets
them to `unknown`. Since `unknown` was not seen during training, the one-hot
metadata features are all zero.

This makes the external evaluation stricter and should be considered when
interpreting results.

## Outputs

- `artifacts/features/nenrus_feature_table.csv`
- `artifacts/reports/predictions.csv`
- `artifacts/reports/overall_metrics.json`
- `artifacts/reports/summary.json`
- `artifacts/reports/by_distance.csv`
- `artifacts/reports/by_drone_type.csv`
- `artifacts/reports/by_drone_type_and_distance.csv`
- `artifacts/reports/analysis.md`
- `artifacts/plots/*.png`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt5/studies/study02/evaluate_nenrus_external.py"
```

