# Study 04: Nenrus Calibration Function

## Problem We Saw

The exported RF+XGBoost ensemble overpredicted every Nenrus image before calibration.
This is a domain-shift problem: the model was trained on the original controlled/synthetic-style dataset, while Nenrus contains real-drone images with different bbox scale behavior.

- raw MAE: `26.1078m`
- raw mean relative error: `1.0002`
- raw overprediction rate: `1.0000`
- raw within 10m rate: `0.0757`

## Selected Calibration Function

`per_drone_quadratic`

- `Kongsberg: corrected = -6.80505587 + 0.65212774 * prediction + -0.00146665 * prediction^2`
- `Vestfold: corrected = -11.71790603 + 1.09969741 * prediction + -0.00305963 * prediction^2`

## Honest Split Evaluation

The repeated split evaluation learns calibration on a small labelled subset of Nenrus and evaluates on the remaining held-out Nenrus rows.

For the selected model at 20% calibration data:

- test MAE mean: `3.0735m`
- test MAE std: `0.0685m`
- mean relative error: `0.1178`
- within 10m rate: `0.9654`

Best 20% calibration models:

- `per_drone_quadratic`: MAE `3.0735m`, relative `0.1178`, within10 `0.9654`
- `per_drone_affine`: MAE `3.4065m`, relative `0.1426`, within10 `0.9443`
- `global_quadratic`: MAE `7.5341m`, relative `0.2642`, within10 `0.6764`
- `global_affine`: MAE `7.7144m`, relative `0.2981`, within10 `0.7009`
- `global_scale_only`: MAE `7.7287m`, relative `0.3044`, within10 `0.6988`

## Full-Data Fit

The final saved function is fitted on all Nenrus rows. This is useful if the goal is to calibrate this specific real-domain dataset, but it is not a pure external-test score.

- full-fit MAE: `3.0251m`
- full-fit mean relative error: `0.1165`
- full-fit within 10m rate: `0.9714`
- full-fit mean signed error: `-0.0000m`

## Per Dataset Result

- `Kongsberg`: MAE `36.1536m` -> `2.7566m`, relative `1.5833` -> `0.1325`
- `Vestfold`: MAE `18.9774m` -> `3.2156m`, relative `0.5864` -> `0.1051`

## Metadata Note

Nenrus does not provide usable `weather` or `time_of_day` values for the trained metadata categories.
The feature pipeline used `unknown`, not null/NaN. Because `unknown` was not in training, the one-hot metadata columns are all zero.
- `weather` observed values: `unknown`; matching trained levels: `0/489`
- `time_of_day` observed values: `unknown`; matching trained levels: `0/489`

## Conclusion

A per-drone quadratic calibration is the strongest simple correction for the systematic overprediction.
The per-drone affine version is slightly weaker but remains a useful simpler fallback.
It does not retrain the RF/XGBoost model; it only maps the model's raw distance output onto the real-domain distance scale.
