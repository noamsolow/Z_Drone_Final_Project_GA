# Study 04: Nenrus Calibration Function

## Problem We Saw

The exported RF+XGBoost ensemble overpredicted every Nenrus image before calibration.
This is a domain-shift problem: the model was trained on the original controlled/synthetic-style dataset, while Nenrus contains real-drone images with different bbox scale behavior.

- raw MAE: `23.9967m`
- raw mean relative error: `0.9407`
- raw overprediction rate: `1.0000`
- raw within 10m rate: `0.1595`

## Selected Calibration Function

`per_drone_quadratic`

- `Kongsberg: corrected = -7.49637448 + 0.69386725 * prediction + -0.00168007 * prediction^2`
- `Vestfold: corrected = -13.57020800 + 1.20957612 * prediction + -0.00380551 * prediction^2`

## Honest Split Evaluation

The repeated split evaluation learns calibration on a small labelled subset of Nenrus and evaluates on the remaining held-out Nenrus rows.

For the selected model at 20% calibration data:

- test MAE mean: `3.2264m`
- test MAE std: `0.0699m`
- mean relative error: `0.1252`
- within 10m rate: `0.9639`

Best 20% calibration models:

- `per_drone_quadratic`: MAE `3.2264m`, relative `0.1252`, within10 `0.9639`
- `per_drone_affine`: MAE `3.6705m`, relative `0.1535`, within10 `0.9310`
- `global_quadratic`: MAE `7.6709m`, relative `0.2705`, within10 `0.6495`
- `global_affine`: MAE `7.8623m`, relative `0.3060`, within10 `0.6835`
- `global_scale_only`: MAE `7.8753m`, relative `0.3116`, within10 `0.6835`

## Full-Data Fit

The final saved function is fitted on all Nenrus rows. This is useful if the goal is to calibrate this specific real-domain dataset, but it is not a pure external-test score.

- full-fit MAE: `3.1812m`
- full-fit mean relative error: `0.1240`
- full-fit within 10m rate: `0.9652`
- full-fit mean signed error: `0.0000m`

## Per Dataset Result

- `Kongsberg`: MAE `34.0035m` -> `2.8003m`, relative `1.5071` -> `0.1302`
- `Vestfold`: MAE `16.8940m` -> `3.4516m`, relative `0.5386` -> `0.1195`

## Metadata Note

Nenrus does not provide usable `weather` or `time_of_day` values for the trained metadata categories.
The feature pipeline used `unknown`, not null/NaN. Because `unknown` was not in training, the one-hot metadata columns are all zero.

## Conclusion

A per-drone quadratic calibration is the strongest simple correction for the systematic overprediction.
The per-drone affine version is slightly weaker but remains a useful simpler fallback.
It does not retrain the RF/XGBoost model; it only maps the model's raw distance output onto the real-domain distance scale.
