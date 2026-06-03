# Attempt 5 Study 02 External Nenrus Analysis

## External Verdict

The exported Attempt 5 ensemble does **not** generalize well to the prepared Nenrus dataset as a ready-to-use model.

The dominant failure mode is systematic overprediction:

- overprediction rate: `1.000`
- underprediction rate: `0.000`
- mean signed error: `26.108m`

This means the model is not just noisy on Nenrus; it is biased upward across the dataset.

## Overall Result

- Count: `489`
- MAE: `26.108m`
- Median absolute error: `21.935m`
- Mean relative error: `1.000`
- RMSE: `30.483m`
- R2: `-1.447`
- Within 5m: `0.002`
- Within 10m: `0.076`
- Mean signed error: `26.108m`

## Main Takeaway

Attempt 4's held-out performance was strong because train/test came from the same original data distribution.
Nenrus is a different visual domain with real-drone images, different bbox scale behavior, shorter distance ranges, and missing weather/time metadata.
The external result therefore exposes a real domain shift.

The model can still be useful diagnostically: it tells us that bbox geometry learned from the original dataset does not transfer cleanly to these real-drone images.
But it should not be treated as a calibrated ready model for Nenrus-style images without an adaptation/calibration step.

## Metadata Note

Nenrus does not provide weather/time_of_day metadata. Both fields were set to `unknown`, which makes all trained metadata one-hot columns zero.

## By Drone Type / Source

- `Kongsberg`: count `203`, MAE `36.154m`, mean signed error `36.154m`, within 10m `0.000`
- `Vestfold`: count `286`, MAE `18.977m`, mean signed error `18.977m`, within 10m `0.129`

## Best Distance Buckets

- `10.0`m: count `19`, MAE `10.184m`, mean signed error `10.184m`
- `60.0`m: count `13`, MAE `13.578m`, mean signed error `13.578m`
- `9.0`m: count `26`, MAE `13.960m`, mean signed error `13.960m`
- `15.0`m: count `44`, MAE `15.118m`, mean signed error `15.118m`
- `4.0`m: count `18`, MAE `15.880m`, mean signed error `15.880m`

## Worst Distance Buckets

- `45.0`m: count `39`, MAE `42.584m`, mean signed error `42.584m`
- `70.0`m: count `12`, MAE `38.059m`, mean signed error `38.059m`
- `75.0`m: count `24`, MAE `37.353m`, mean signed error `37.353m`
- `50.0`m: count `42`, MAE `36.608m`, mean signed error `36.608m`
- `40.0`m: count `37`, MAE `33.346m`, mean signed error `33.346m`

## Worst Drone-Type x Distance Cells

- `Kongsberg` `45.0`m: count `20`, MAE `59.326m`, mean signed error `59.326m`
- `Kongsberg` `35.0`m: count `21`, MAE `51.861m`, mean signed error `51.861m`
- `Kongsberg` `50.0`m: count `21`, MAE `51.375m`, mean signed error `51.375m`
- `Kongsberg` `40.0`m: count `21`, MAE `46.855m`, mean signed error `46.855m`
- `Kongsberg` `30.0`m: count `19`, MAE `38.829m`, mean signed error `38.829m`
- `Vestfold` `70.0`m: count `12`, MAE `38.059m`, mean signed error `38.059m`
- `Vestfold` `75.0`m: count `24`, MAE `37.353m`, mean signed error `37.353m`
- `Kongsberg` `25.0`m: count `16`, MAE `34.334m`, mean signed error `34.334m`

## Initial Interpretation

This is an external-domain test, not a same-distribution held-out test.
The model was trained on synthetic/controlled drone imagery with known weather and time metadata, while Nenrus contains real-drone imagery and does not provide those metadata fields.
If the MAE is substantially higher than Attempt 4, that does not mean the previous model was invalid; it means the domain shift is real.

The key question for the next step is whether the error pattern is systematic enough to correct without retraining on the external test set.
Useful signs would be consistent bias by distance, consistent bias by drone type, or a predictable failure at near/far ranges.

In this run the bias is systematic: predictions are consistently too high.
That suggests the next useful test is not another model-family comparison, but a calibration/domain-adaptation study designed explicitly for external real-drone imagery.
