# Attempt6 Study01: Geometry-Regularized Synthetic-to-Real Generalization

This study asks whether we can improve real-world generalization without fitting
on real images.

The training data is synthetic only. The real Nenrus data is used only as an
external test set.

## Motivation

Previous studies showed that the strongest models rely heavily on bbox geometry,
especially bbox width and area. That gives strong synthetic performance, but it
also creates a shortcut: a different drone body size can look like a different
distance.

The goal here is to keep the strong RF+XGB ensemble structure, but make geometry
less perfect during synthetic training.

## Model Family

The study keeps the model structure that worked best:

- Random Forest
- XGBoost
- weighted RF/XGB blend

The default blend weights are:

- RF: `0.44`
- XGB: `0.56`

## Training and Testing

Training:

- synthetic Attempt4 noisy expanded feature table
- synthetic dev split only

Testing:

- synthetic holdout split
- real Nenrus external data

The real rows are never used for fitting, categorical levels, geometry medians,
dropout parameters, or feature transforms.

## Variants

- `baseline_full_geometry`: normal full feature baseline
- `geometry_dropout_030`: random geometry replacement with train medians
- `geometry_dropout_060`: stronger geometry dropout
- `geometry_noise_020`: multiplicative geometry noise during training
- `geometry_dropout_030_noise_020`: dropout and noise together
- `reduced_geometry`: compact geometry only
- `raw_depth_only`: raw relative-depth-only stress test

The real Nenrus feature table currently contains only the raw bbox-only depth
feature, not the full multiscale depth feature set. For that reason, this study
keeps all external-test variants on features that exist in both synthetic and
real feature tables.

## Outputs

Reports:

- `artifacts/reports/variant_metrics.csv`
- `artifacts/reports/predictions.csv`
- `artifacts/reports/real_by_drone_type.csv`
- `artifacts/reports/real_by_distance.csv`
- `artifacts/reports/feature_importances.csv`
- `artifacts/reports/feature_bucket_importances.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/analysis.md`

Plots:

- `artifacts/plots/real_external_overall_by_variant.png`
- `artifacts/plots/synthetic_vs_real_mae_by_variant.png`
- `artifacts/plots/real_mae_by_drone_type_and_variant.png`
- `artifacts/plots/feature_bucket_importances_by_variant.png`
- `artifacts/plots/winner_real_true_vs_predicted.png`

## Run

Dry run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study01/run_study_01.py" --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study01/run_study_01.py"
```
