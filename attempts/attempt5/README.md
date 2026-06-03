# Attempt 5: Saved Ensemble, Real-Drone Evaluation, and Calibration

Attempt 5 turns the best `attempt4` model into reusable saved artifacts, tests it
on the real-drone Nenrus dataset, diagnoses why it fails out-of-domain, and then
learns a simple calibration layer that makes the predictions usable on Nenrus.

The important high-level result:

- raw exported RF+XGBoost ensemble on Nenrus: `26.1078m` MAE
- after per-drone quadratic calibration: `3.0251m` MAE
- raw model overpredicted `100%` of Nenrus images
- calibrated model is within `10m` on `97.14%` of Nenrus images

## Study Map

- `study01`: export fitted Random Forest, XGBoost, ensemble metadata, and schema.
- `study02`: run the frozen ensemble on the prepared Nenrus real-drone dataset.
- `study03`: test whether making relative-depth more dominant solves the issue.
- `study04`: learn and evaluate calibration functions for Nenrus predictions.

## Study 01: Exported RF+XGBoost Ensemble

Study 01 saves the best model family from `attempt4`:

- RF base model: `raw_depth_geometry_metadata_rf`
- RF candidate: `rf_unbounded`
- XGBoost base model: `raw_depth_geometry_metadata_xgb`
- XGBoost candidate: `xgb_deep`
- tuned blend: `0.44 * RF + 0.56 * XGBoost`
- equal blend reference: `0.50 * RF + 0.50 * XGBoost`

The exported model uses `13` tabular features:

- 1 raw relative-depth feature
- 8 bbox geometry features
- 4 metadata one-hot features

Saved full-dataset model artifacts:

- `studies/study01/artifacts/models/full_dataset/rf_model.joblib`
- `studies/study01/artifacts/models/full_dataset/xgb_model.joblib`
- `studies/study01/artifacts/models/full_dataset/ensemble.joblib`

`ensemble.joblib` is a small metadata artifact. It stores paths to the RF/XGB
models, blend weights, feature schema, and training scope. It does not duplicate
the large model objects.

The `dev_holdout` export reproduced the known Attempt 4 result:

- tuned blend test MAE: `7.6347m`
- equal blend test MAE: `7.6325m`

## Study 02: External Nenrus Evaluation

Study 02 uses the frozen full-dataset ensemble on the prepared Nenrus dataset.
No RF/XGBoost retraining happens here.

Raw external result:

- count: `489`
- MAE: `26.1078m`
- median absolute error: `21.9354m`
- RMSE: `30.4831m`
- R2: `-1.4467`
- mean relative error: `1.0002`
- overprediction rate: `100%`
- within `5m`: `0.20%`
- within `10m`: `7.57%`

By real-drone dataset:

- `Kongsberg`: MAE `36.1536m`, count `203`
- `Vestfold`: MAE `18.9774m`, count `286`

Conclusion: the saved ensemble does not generalize directly to Nenrus-style
real-drone images. It systematically predicts distances that are too large.

## The Issue We Found

The issue is mostly domain shift, not a broken bbox.

At the same true distance, Nenrus drone bounding boxes are much smaller than the
boxes in the original training dataset. The model learned that smaller bbox
geometry usually means the drone is farther away, so it interprets real Nenrus
drones as too far.

Important bbox-domain findings at shared distances:

- Kongsberg bbox width is often only about `44%` to `60%` of the original bbox width.
- Kongsberg bbox area is often only about `12%` to `26%` of the original bbox area.
- Vestfold is closer than Kongsberg, but still often smaller than the original dataset.
- The model relies strongly on bbox geometry, so this scale mismatch creates systematic overprediction.

We also checked random bbox visualizations from both Nenrus datasets. The boxes
looked correct/tight around the drones, so the problem is not simply bad labels.

## Metadata Behavior

The original model was trained with metadata features:

- `weather=clear_sky`
- `weather=light_rain`
- `time_of_day=10AM`
- `time_of_day=8PM`

Nenrus does not provide usable `weather` or `time_of_day` values. The feature
pipeline did not use null/NaN. It set both fields to `unknown`.

Because `unknown` was not seen during training, all trained metadata one-hot
columns become zero for Nenrus:

- `weather`: observed `unknown`, matching trained levels `0/489`
- `time_of_day`: observed `unknown`, matching trained levels `0/489`
- metadata one-hot active rate: `0.0`

This was probably not the main error source. The dominant issue was bbox/domain
shift. Still, it means the model has less context on Nenrus than it had during
training.

Metadata audit:

- [metadata_missing_audit.json](studies/study04/artifacts/reports/metadata_missing_audit.json)
- [metadata_missing_one_hot_audit.png](studies/study04/artifacts/plots/metadata_missing_one_hot_audit.png)

## Study 03: Relative-Depth Weighting Test

Study 03 tested whether giving relative depth more influence could solve the
Nenrus failure.

Variants included:

- `depth_only`
- `depth_plus_no_size_geometry`
- `depth_plus_normalized_geometry`
- `depth_repeated_8_with_geometry`
- `depth_repeated_16_with_geometry`

Best variant:

- `depth_plus_normalized_geometry`: MAE `21.611m`

Baseline saved ensemble:

- MAE `26.108m`

Conclusion: emphasizing relative depth helped somewhat, but it did not solve the
problem. The best depth-heavy variant still overpredicted every Nenrus sample.
Calibration is still needed.

Study 03 report:

- [Study 03 README](studies/study03/README.md)
- [Study 03 analysis](studies/study03/artifacts/reports/analysis.md)

## Study 04: Calibration Solution

Study 04 learns a small correction layer on top of the raw ensemble prediction.
It does not retrain RF or XGBoost.

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

Why per-drone calibration works better:

- Kongsberg and Vestfold have different bbox scale behavior.
- A single global correction helps, but it cannot fully represent both domains.
- Separate functions let each real-drone subset map raw predictions back to the correct distance scale.

## Honest Calibration Evaluation

The honest split evaluation learns calibration on a small labelled subset of
Nenrus and evaluates on the remaining held-out Nenrus rows.

At `20%` calibration data:

- `per_drone_quadratic`: MAE `3.0735m`, relative error `0.1178`, within10 `0.9654`
- `per_drone_affine`: MAE `3.4065m`, relative error `0.1426`, within10 `0.9443`
- `global_quadratic`: MAE `7.5341m`, relative error `0.2642`, within10 `0.6764`
- `global_affine`: MAE `7.7144m`, relative error `0.2981`, within10 `0.7009`
- `global_scale_only`: MAE `7.7287m`, relative error `0.3044`, within10 `0.6988`

This is the most honest estimate of what happens if we label a small calibration
set from the real domain and then test on the rest.

## Final Full-Data Calibration Fit

The final saved calibration function is fitted on all Nenrus rows. This is useful
for calibrating this specific real-domain dataset, but it is not a pure external
test score.

Final full-data result:

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
- overprediction rate: `47.85%`
- underprediction rate: `52.15%`

Before/after by dataset:

- `Kongsberg`: MAE `36.1536m` -> `2.7566m`
- `Vestfold`: MAE `18.9774m` -> `3.2156m`

## Graphs That Explain the Problem and Solution

Main calibration graph package:

- [raw_vs_calibrated_predicted_vs_true.png](studies/study04/artifacts/plots/raw_vs_calibrated_predicted_vs_true.png)
- [calibration_mapping_curve.png](studies/study04/artifacts/plots/calibration_mapping_curve.png)
- [mae_before_after_by_distance.png](studies/study04/artifacts/plots/mae_before_after_by_distance.png)
- [mae_before_after_by_drone_type.png](studies/study04/artifacts/plots/mae_before_after_by_drone_type.png)
- [error_distribution_before_after.png](studies/study04/artifacts/plots/error_distribution_before_after.png)
- [absolute_error_cdf_before_after.png](studies/study04/artifacts/plots/absolute_error_cdf_before_after.png)
- [tolerance_rates_before_after.png](studies/study04/artifacts/plots/tolerance_rates_before_after.png)
- [relative_error_before_after_by_distance.png](studies/study04/artifacts/plots/relative_error_before_after_by_distance.png)
- [residuals_after_calibration.png](studies/study04/artifacts/plots/residuals_after_calibration.png)
- [mean_prediction_tracks_by_distance.png](studies/study04/artifacts/plots/mean_prediction_tracks_by_distance.png)
- [metadata_missing_one_hot_audit.png](studies/study04/artifacts/plots/metadata_missing_one_hot_audit.png)
- [calibration_split_performance.png](studies/study04/artifacts/plots/calibration_split_performance.png)
- [final_calibrated_predicted_vs_true.png](studies/study04/artifacts/plots/final_calibrated_predicted_vs_true.png)
- [final_calibrated_mae_by_distance.png](studies/study04/artifacts/plots/final_calibrated_mae_by_distance.png)

Useful CSV/JSON outputs:

- [final_calibration_functions.json](studies/study04/artifacts/reports/final_calibration_functions.json)
- [raw_uncalibrated_metrics.json](studies/study04/artifacts/reports/raw_uncalibrated_metrics.json)
- [final_calibrated_metrics.json](studies/study04/artifacts/reports/final_calibrated_metrics.json)
- [before_after_by_drone_type.csv](studies/study04/artifacts/reports/before_after_by_drone_type.csv)
- [before_after_by_distance.csv](studies/study04/artifacts/reports/before_after_by_distance.csv)
- [split_eval_summary.csv](studies/study04/artifacts/reports/split_eval_summary.csv)
- [analysis.md](studies/study04/artifacts/reports/analysis.md)

## Final Conclusion

The exported Attempt 4 ensemble is strong on the original dataset, but it is not
directly calibrated for real-drone Nenrus images. The failure is systematic:
real-drone bbox geometry is smaller/different, so the model predicts distances
that are too far.

The best practical solution so far is not retraining the RF/XGBoost model. It is
to keep the frozen ensemble and add a small per-drone calibration function on top
of its raw prediction.

With only `20%` labelled Nenrus calibration data, the calibrated model reaches
about `3.07m` held-out MAE. With the final function fitted on all Nenrus labels,
the dataset-level MAE is `3.0251m`.
