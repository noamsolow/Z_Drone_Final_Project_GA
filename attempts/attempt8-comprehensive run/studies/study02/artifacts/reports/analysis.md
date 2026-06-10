# Study 04: Nenrus Calibration Function

## Problem We Saw

The exported RF+XGBoost ensemble overpredicted every Nenrus image before calibration.
This is a domain-shift problem: the model was trained on the original controlled/synthetic-style dataset, while Nenrus contains real-drone images with different bbox scale behavior.

- raw MAE: `23.7959m`
- raw mean relative error: `0.9337`
- raw overprediction rate: `0.9980`
- raw within 10m rate: `0.1595`

## Selected Calibration Function

`per_drone_quadratic`

- `Kongsberg: corrected = -7.70205091 + 0.70694685 * prediction + -0.00179344 * prediction^2`
- `Vestfold: corrected = -14.40928355 + 1.25810371 * prediction + -0.00421957 * prediction^2`

## Honest Split Evaluation

The repeated split evaluation learns calibration on a small labelled subset of Nenrus and evaluates on the remaining held-out Nenrus rows.

For the selected model at 20% calibration data:

- test MAE mean: `3.2781m`
- test MAE std: `0.0727m`
- mean relative error: `0.1246`
- within 10m rate: `0.9644`

Best 20% calibration models:

- `per_drone_quadratic`: MAE `3.2781m`, relative `0.1246`, within10 `0.9644`
- `per_drone_affine`: MAE `3.8726m`, relative `0.1617`, within10 `0.9197`
- `global_quadratic`: MAE `7.6897m`, relative `0.2720`, within10 `0.6477`
- `global_scale_only`: MAE `7.8899m`, relative `0.3127`, within10 `0.6850`
- `global_affine`: MAE `7.8967m`, relative `0.3128`, within10 `0.6859`

## Full-Data Fit

The final saved function is fitted on all Nenrus rows. This is useful if the goal is to calibrate this specific real-domain dataset, but it is not a pure external-test score.

- full-fit MAE: `3.2242m`
- full-fit mean relative error: `0.1228`
- full-fit within 10m rate: `0.9673`
- full-fit mean signed error: `-0.0000m`

## Per Dataset Result

- `Kongsberg`: MAE `33.8633m` -> `2.7849m`, relative `1.5011` -> `0.1278`
- `Vestfold`: MAE `16.6501m` -> `3.5359m`, relative `0.5309` -> `0.1193`

## Metadata Note

Nenrus does not provide usable `weather` or `time_of_day` values for the trained metadata categories.
The feature pipeline used `unknown`, not null/NaN. Because `unknown` was not in training, the one-hot metadata columns are all zero.

## Conclusion

A per-drone quadratic calibration is the strongest simple correction for the systematic overprediction.
The per-drone affine version is slightly weaker but remains a useful simpler fallback.
It does not retrain the RF/XGBoost model; it only maps the model's raw distance output onto the real-domain distance scale.
