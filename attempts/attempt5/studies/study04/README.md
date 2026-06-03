# Study 04: Nenrus Calibration Function

This study learns a small calibration layer on top of the exported model's
Nenrus predictions.

It does not retrain the RF or XGBoost models.

The short result:

- raw exported ensemble on Nenrus: `26.1078m` MAE
- calibrated ensemble on Nenrus: `3.0251m` MAE
- raw overprediction rate: `100%`
- calibrated within `10m` rate: `97.14%`

## Goal

The model's raw Nenrus predictions were systematically too high.

This study asks:

- can we learn a simple function from `predicted_distance` to `true_distance`?
- does a small labelled calibration subset fix the bias?
- should the function be global or separate per drone type?

## Calibration Functions Tested

- `raw_uncalibrated`
- `global_affine`
- `global_quadratic`
- `global_scale_only`
- `per_drone_affine`
- `per_drone_quadratic`

The selected function is `per_drone_quadratic`. Let `p` be the raw ensemble
prediction.

Kongsberg:

```text
calibrated = -6.80505587275528 + 0.6521277411976535 * p - 0.001466649376865686 * p^2
```

Vestfold:

```text
calibrated = -11.71790602892214 + 1.0996974109541213 * p - 0.003059627825357708 * p^2
```

## Problem Diagnosis

The raw model failed because of domain shift.

The original model learned from controlled/synthetic-style images. Nenrus
contains real-drone images where, at the same metric distance, the drone bboxes
are often much smaller than in the original dataset. Since the model relies on
bbox geometry, it reads those smaller boxes as "farther away" and overpredicts.

This was not mainly a bbox-label bug. Random bbox visual checks showed the Nenrus
boxes were correctly placed around the drones.

## Metadata Behavior

Nenrus does not provide usable `weather` or `time_of_day` metadata.

The pipeline used `unknown`, not null/NaN. Because `unknown` was not a trained
category, all trained metadata one-hot columns were zero:

- `weather`: observed `unknown`, matching trained levels `0/489`
- `time_of_day`: observed `unknown`, matching trained levels `0/489`
- metadata one-hot active rate: `0.0`

This likely reduced context, but the dominant issue was bbox/domain shift.

## Honest Split Result

At `20%` labelled calibration data, evaluated on held-out Nenrus rows:

- `per_drone_quadratic`: MAE `3.0735m`, relative error `0.1178`, within10 `0.9654`
- `per_drone_affine`: MAE `3.4065m`, relative error `0.1426`, within10 `0.9443`
- `global_quadratic`: MAE `7.5341m`, relative error `0.2642`, within10 `0.6764`
- `global_affine`: MAE `7.7144m`, relative error `0.2981`, within10 `0.7009`
- `global_scale_only`: MAE `7.7287m`, relative error `0.3044`, within10 `0.6988`

## Final Full-Data Fit

The final saved calibration function is fitted on all Nenrus rows. This is useful
for this dataset, but it is not a pure external-test score.

Final full-data metrics:

- count: `489`
- MAE: `3.0251m`
- median absolute error: `2.0153m`
- mean relative error: `0.1165`
- RMSE: `4.1104m`
- R2: `0.9555`
- within `2m`: `49.28%`
- within `5m`: `79.75%`
- within `10m`: `97.14%`
- within `20m`: `100%`
- mean signed error: approximately `0m`

Before/after by dataset:

- `Kongsberg`: MAE `36.1536m` -> `2.7566m`
- `Vestfold`: MAE `18.9774m` -> `3.2156m`

## Graphs

These plots explain the issue and the fix:

- `artifacts/plots/raw_vs_calibrated_predicted_vs_true.png`
- `artifacts/plots/calibration_mapping_curve.png`
- `artifacts/plots/mae_before_after_by_distance.png`
- `artifacts/plots/mae_before_after_by_drone_type.png`
- `artifacts/plots/error_distribution_before_after.png`
- `artifacts/plots/absolute_error_cdf_before_after.png`
- `artifacts/plots/tolerance_rates_before_after.png`
- `artifacts/plots/relative_error_before_after_by_distance.png`
- `artifacts/plots/residuals_after_calibration.png`
- `artifacts/plots/mean_prediction_tracks_by_distance.png`
- `artifacts/plots/metadata_missing_one_hot_audit.png`
- `artifacts/plots/calibration_split_performance.png`
- `artifacts/plots/final_calibrated_predicted_vs_true.png`
- `artifacts/plots/final_calibrated_mae_by_distance.png`

## Outputs

- `artifacts/reports/split_eval_runs.csv`
- `artifacts/reports/split_eval_summary.csv`
- `artifacts/reports/final_calibration_functions.json`
- `artifacts/reports/raw_uncalibrated_metrics.json`
- `artifacts/reports/final_calibrated_predictions.csv`
- `artifacts/reports/final_calibrated_metrics.json`
- `artifacts/reports/before_after_by_drone_type.csv`
- `artifacts/reports/before_after_by_distance.csv`
- `artifacts/reports/metadata_missing_audit.json`
- `artifacts/reports/analysis.md`
- `artifacts/plots/*.png`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt5/studies/study04/run_nenrus_calibration.py"
```
