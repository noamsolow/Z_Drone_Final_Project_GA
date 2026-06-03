# Attempt 4: Noisy-BBox Expanded Rows, XGBoost, and RF+XGB Blending

This document is the full recap of `attempt4`.

It explains:

- why this attempt was created after `attempt3`
- what each study tested
- how the noisy-bbox expanded-row dataset was built
- what models were compared
- what the results were
- how to interpret the results
- what we learned about relative depth
- what the best current model is
- what is still missing before true external inference

In short, `attempt4` moved from the aggregated noisy-bbox representation of
`attempt3` to a more direct detector-like representation:

- each source image is expanded into several bbox jitter rows
- each jitter row is treated as a real prediction case
- rows from the same source image are grouped together in train/dev/test splits
- Random Forest and XGBoost are compared on the exact same feature table
- the best RF and XGBoost predictions are blended into an ensemble

The best current result from this attempt is the RF+XGBoost blend:

- best CV-selected blend: `tuned_weight_blend`
- RF weight: `0.44`
- XGBoost weight: `0.56`
- CV OOF MAE: `7.6770m`
- test MAE: `7.6347m`

On the held-out test set, the equal 50/50 blend was fractionally better:

- equal blend test MAE: `7.6325m`

The difference is tiny, so the practical conclusion is:

- the strongest model family from `attempt4` is an RF+XGBoost ensemble
- the exact blend weight is not very sensitive around 50/50

## 1. Why Attempt 4 Exists

`attempt3` showed that noisy-bbox robustness matters.

In `attempt3/study03`, we simulated bbox noise by jittering each bounding box,
computing geometry and depth features for each jitter, and then aggregating the
jittered values back into one robust row per source image.

That was useful, but it still answered a slightly abstract question:

- how good is a robust per-image representation after aggregating bbox noise?

`attempt4` asks a more detector-like question:

- if a detector gives us a slightly imperfect bbox, can the distance model still
  perform well on that individual imperfect detection?

So instead of aggregating jittered boxes back into one row, `attempt4` keeps
each jitter as its own row.

This makes the task harder and more realistic:

- the same original image can produce several possible bbox observations
- geometry changes across jitter rows
- relative depth changes across jitter rows
- the model must tolerate those row-level variations

## 2. High-Level Structure

`attempt4` contains three studies:

- [Study 01](studies/study01): noisy-bbox expanded rows with Random Forest
- [Study 02](studies/study02): same feature table with XGBoost
- [Study 03](studies/study03): blend the best RF and XGBoost predictions

The studies form a clean progression:

- Study 01 tests whether the expanded noisy-bbox setup works with RF.
- Study 02 keeps the same data construction and swaps the learner to XGBoost.
- Study 03 asks whether RF and XGBoost make complementary errors.

## 3. Dataset Construction

The base dataset is the same drone image dataset used in the previous attempts.

From the Study 01 feature summary:

- source images: `15,064`
- augmented rows: `60,256`
- jitter rows per source image: `4`
- distance values: `20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 125, 150`
- weather values: `clear_sky`, `light_rain`
- time-of-day values: `10AM`, `8PM`
- strata: `48`

The four jitter variants are:

- `original`
- `scale_down`
- `scale_up`
- `scale_up_more`

This version is scale-only. It does not add directional left/right/up/down
shifts.

## 4. Split-Safety Rule

The most important methodological rule in `attempt4` is that all jitter rows
from the same source image stay in the same split group.

That prevents leakage.

Without that rule, the model could see one jittered version of an image during
training and another jittered version of the same image during test. That would
make the test result artificially optimistic.

The actual split sizes in Study 01 and Study 02 were:

- total rows: `60,256`
- dev rows: `51,216`
- test rows: `9,040`
- unique source images: `15,064`

## 5. Feature Families

Each jitter row contains depth, geometry, and metadata features.

### Depth Contexts

Depth is computed in several bbox contexts:

- `bbox_only`
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- `bbox_expand_4x`

### Depth Aggregation Methods

For each bbox context, the study computes:

- `bbox_midpoint`
- `bbox_mean`
- `inner50_median`

That creates `12` raw relative-depth features.

The selected raw depth feature used by the full models is:

- `bbox_only__inner50_median__object_depth`

### Geometry Features

The geometry feature family includes:

- bbox width in pixels
- bbox height in pixels
- normalized bbox width
- normalized bbox height
- bbox area ratio
- bbox aspect ratio
- normalized center x
- normalized center y

### Metadata Features

The metadata family includes:

- weather one-hot features
- time-of-day one-hot features

## 6. Models Compared

Study 01 and Study 02 use the same comparison structure.

Each study compares:

- depth-only model
- geometry+metadata model
- raw-depth+geometry+metadata model
- small learned-depth-only model
- stacked learned-depth+geometry+metadata model

The goal was not only to find the best model, but also to isolate what each
feature family contributes.

## 7. Study 01: Random Forest on Expanded Rows

Main files:

- [run_study_01.py](studies/study01/run_study_01.py)
- [config.yaml](studies/study01/config.yaml)
- [study README](studies/study01/README.md)
- [summary.json](studies/study01/artifacts/reports/summary.json)
- [test_metrics.json](studies/study01/artifacts/reports/test_metrics.json)

### Goal

Study 01 asked:

- if we stop aggregating jittered boxes,
- and instead keep each noisy bbox as its own row,
- how well does Random Forest handle the resulting detector-like noise?

### Models

The Random Forest candidates included:

- `rf_shallow`
- `rf_unbounded`

The winning upper candidate was:

- `rf_unbounded`

### Main Result

Winner:

- `raw_depth_geometry_metadata_rf`

From [summary.json](studies/study01/artifacts/reports/summary.json):

- CV OOF MAE: `7.7907m`
- CV mean relative error: `0.1010`
- test MAE: `7.7124m`
- test mean relative error: `0.0990`

### Study 01 Test Results

| Model | Candidate | Test MAE | Mean Relative Error | RMSE | R2 | Within 10m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `raw_depth_only_rf` | `rf_shallow` | `26.7110` | `0.5140` | `33.3592` | `0.2368` | `0.2537` |
| `geometry_metadata_rf` | `rf_unbounded` | `8.0335` | `0.1065` | `11.2489` | `0.9132` | `0.7002` |
| `raw_depth_geometry_metadata_rf` | `rf_unbounded` | `7.7124` | `0.0990` | `10.8639` | `0.9191` | `0.7148` |
| `small_rf_depth_only_rf` | `rf_shallow` | `20.6553` | `0.3548` | `26.6844` | `0.5117` | `0.3374` |
| `stacked_rf_depth_geometry_metadata_rf` | `rf_unbounded` | `8.0300` | `0.1048` | `11.1279` | `0.9151` | `0.6955` |

### What Study 01 Means

The RF result is very clear:

- depth-only is not strong enough by itself
- geometry+metadata is already very strong
- adding the raw relative-depth scalar improves the full model
- the stacked learned-depth feature does not beat the simpler raw-depth feature

The key improvement from relative depth in RF was:

- `geometry_metadata_rf`: `8.0335m`
- `raw_depth_geometry_metadata_rf`: `7.7124m`
- improvement: `0.3211m MAE`

That is not a huge effect, but it is real and useful.

## 8. Study 02: XGBoost on the Same Expanded Rows

Main files:

- [run_study_02.py](studies/study02/run_study_02.py)
- [config.yaml](studies/study02/config.yaml)
- [study README](studies/study02/README.md)
- [summary.json](studies/study02/artifacts/reports/summary.json)
- [test_metrics.json](studies/study02/artifacts/reports/test_metrics.json)

### Goal

Study 02 asked:

- if we keep the exact same feature table from Study 01,
- but replace Random Forest with XGBoost,
- do we get a stronger learner?

This was an important controlled comparison because it changed the model family
without changing the data construction.

### Models

The XGBoost candidates included:

- `xgb_shallow`
- `xgb_deep`

The winning upper candidate was:

- `xgb_deep`

### Main Result

Winner:

- `raw_depth_geometry_metadata_xgb`

From [summary.json](studies/study02/artifacts/reports/summary.json):

- CV OOF MAE: `7.7536m`
- CV mean relative error: `0.1028`
- test MAE: `7.7301m`
- test mean relative error: `0.1014`

### Study 02 Test Results

| Model | Candidate | Test MAE | Mean Relative Error | RMSE | R2 | Within 10m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `raw_depth_only_xgb` | `xgb_shallow` | `26.6562` | `0.5135` | `33.3180` | `0.2387` | `0.2549` |
| `geometry_metadata_xgb` | `xgb_deep` | `8.1890` | `0.1105` | `11.1100` | `0.9154` | `0.6869` |
| `raw_depth_geometry_metadata_xgb` | `xgb_deep` | `7.7301` | `0.1014` | `10.6617` | `0.9220` | `0.7140` |
| `small_xgb_depth_only_xgb` | `xgb_shallow` | `20.9286` | `0.3689` | `26.7620` | `0.5088` | `0.3280` |
| `stacked_xgb_depth_geometry_metadata_xgb` | `xgb_deep` | `7.9000` | `0.1065` | `10.7141` | `0.9213` | `0.7028` |

### What Study 02 Means

XGBoost behaved similarly to RF:

- depth-only is weak
- geometry+metadata is strong
- raw relative depth improves the full model
- stacked learned depth is not the winner

The key improvement from relative depth in XGBoost was:

- `geometry_metadata_xgb`: `8.1890m`
- `raw_depth_geometry_metadata_xgb`: `7.7301m`
- improvement: `0.4589m MAE`

This means relative depth helped XGBoost even more than it helped RF in this
setup.

## 9. Study 03: RF + XGBoost Blend

Main files:

- [run_study_03.py](studies/study03/run_study_03.py)
- [config.yaml](studies/study03/config.yaml)
- [study README](studies/study03/README.md)
- [summary.json](studies/study03/artifacts/reports/summary.json)
- [test_metrics.json](studies/study03/artifacts/reports/test_metrics.json)
- [blend_search.csv](studies/study03/artifacts/reports/blend_search.csv)

### Goal

Study 03 asked:

- RF and XGBoost are close in overall MAE,
- but do they make different enough errors that averaging them helps?

This study did not retrain the base models. It loaded the winner OOF and test
predictions from Study 01 and Study 02, aligned them by `augmented_row_id`, and
searched for the best convex blend weight on OOF predictions.

### Models Compared

- `rf_winner_baseline`
- `xgb_winner_baseline`
- `equal_weight_blend`
- `tuned_weight_blend`

### Main Result

CV-selected winner:

- `tuned_weight_blend`

Blend weight:

- RF: `0.44`
- XGBoost: `0.56`

From [summary.json](studies/study03/artifacts/reports/summary.json):

- CV OOF MAE: `7.6770m`
- CV mean relative error: `0.1005`
- test MAE: `7.6347m`
- test mean relative error: `0.0990`

### Study 03 Test Results

| Model | RF Weight | XGB Weight | Test MAE | Mean Relative Error | RMSE | R2 | Within 10m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rf_winner_baseline` | `1.00` | `0.00` | `7.7124` | `0.0990` | `10.8639` | `0.9191` | `0.7148` |
| `xgb_winner_baseline` | `0.00` | `1.00` | `7.7301` | `0.1014` | `10.6617` | `0.9220` | `0.7140` |
| `equal_weight_blend` | `0.50` | `0.50` | `7.6325` | `0.0988` | `10.6396` | `0.9224` | `0.7171` |
| `tuned_weight_blend` | `0.44` | `0.56` | `7.6347` | `0.0990` | `10.6291` | `0.9225` | `0.7162` |

### What Study 03 Means

The blend improved over both individual base models.

Compared with RF alone:

- RF test MAE: `7.7124m`
- tuned blend test MAE: `7.6347m`
- improvement: `0.0777m MAE`

Compared with XGBoost alone:

- XGBoost test MAE: `7.7301m`
- tuned blend test MAE: `7.6347m`
- improvement: `0.0955m MAE`

The equal blend was fractionally better on test:

- equal blend: `7.6325m`
- tuned blend: `7.6347m`
- difference: `0.0022m MAE`

That difference is too small to over-interpret. The stronger conclusion is that
RF and XGBoost complement each other, and that a simple near-50/50 blend is a
good candidate for the final model family.

## 10. Overall Attempt 4 Results

| Study | Winner | Candidate / Weight | CV OOF MAE | Test MAE | Test Mean Relative Error |
| --- | --- | --- | ---: | ---: | ---: |
| Study 01 RF | `raw_depth_geometry_metadata_rf` | `rf_unbounded` | `7.7907` | `7.7124` | `0.0990` |
| Study 02 XGBoost | `raw_depth_geometry_metadata_xgb` | `xgb_deep` | `7.7536` | `7.7301` | `0.1014` |
| Study 03 Blend | `tuned_weight_blend` | `RF=0.44, XGB=0.56` | `7.6770` | `7.6347` | `0.0990` |

Best by CV:

- `tuned_weight_blend`

Best by held-out test MAE:

- `equal_weight_blend`

Practical best model family:

- RF+XGBoost ensemble over `raw_depth_geometry_metadata`

## 11. What We Learned About Relative Depth

Relative depth helped, but it did not dominate the model.

The strongest signal is still bbox geometry.

### RF Relative-Depth Effect

| Comparison | Test MAE |
| --- | ---: |
| `geometry_metadata_rf` | `8.0335` |
| `raw_depth_geometry_metadata_rf` | `7.7124` |
| improvement from raw relative depth | `0.3211` |

### XGBoost Relative-Depth Effect

| Comparison | Test MAE |
| --- | ---: |
| `geometry_metadata_xgb` | `8.1890` |
| `raw_depth_geometry_metadata_xgb` | `7.7301` |
| improvement from raw relative depth | `0.4589` |

So the answer is:

- yes, relative depth matters
- no, relative depth is not the main driver
- geometry remains the dominant predictor
- relative depth is best used as an extra correction signal

## 12. Feature Importance Interpretation

The winner RF model's top features were dominated by bbox geometry:

| RF Feature | Importance |
| --- | ---: |
| `bbox_width_norm` | `0.2888` |
| `bbox_width_px` | `0.2169` |
| `bbox_area_ratio` | `0.1864` |
| `bbox_height_norm` | `0.1074` |
| `bbox_height_px` | `0.0947` |
| `bbox_only__inner50_median__object_depth` | `0.0430` |

The winner XGBoost model showed an even stronger geometry dominance:

| XGBoost Feature | Importance |
| --- | ---: |
| `bbox_width_norm` | `0.6725` |
| `bbox_width_px` | `0.1911` |
| `bbox_area_ratio` | `0.0435` |
| `bbox_only__inner50_median__object_depth` | `0.0130` |
| `time_of_day=8PM` | `0.0120` |
| `bbox_height_norm` | `0.0120` |

This matches the performance results:

- depth-only models are weak
- geometry models are strong
- depth improves geometry models, but only as a secondary signal

## 13. Distance-Range Behavior

The winner models perform best on near ranges and become harder to trust at far
ranges.

For the tuned blend:

- near MAE: `4.2763m`
- mid MAE: `8.8604m`
- far MAE: `11.6445m`

This is expected.

At far distances:

- the drone is smaller in the image
- bbox geometry changes less per meter
- depth estimates become less discriminative
- small bbox errors have larger metric-distance consequences

The model also tends to underpredict far distances:

- tuned blend far mean signed error: `-5.3715m`
- tuned blend far underprediction rate: `0.6600`

So a future improvement could be a far-distance correction layer or a
distance-range-aware model.

## 14. Comparison With Attempt 3

`attempt3/study03` had a lower test MAE:

- Attempt 3 Study 03 winner test MAE: `5.2105m`
- Attempt 4 Study 03 tuned blend test MAE: `7.6347m`

But this should not be read as a simple regression.

The task changed:

- Attempt 3 aggregated jittered bboxes into one robust row per image.
- Attempt 4 keeps each jittered bbox as a separate row.
- Attempt 3 test rows: `2,080`.
- Attempt 4 test rows: `9,040`.

So Attempt 4 is a more direct noisy-detection stress test. Its absolute MAE is
higher, but it answers a harder question.

The useful conclusion is:

- aggregated robust features are easier and cleaner
- expanded noisy rows better simulate detector output
- the model remains reasonably strong under that harder setup

## 15. Current Limitation: No Saved Fitted Models

The studies saved reports, predictions, metrics, feature tables, and feature
importances.

They did not save fitted model artifacts.

Currently saved:

- `summary.json`
- `cv_metrics.json`
- `test_metrics.json`
- `cv_predictions.csv`
- `test_predictions.csv`
- feature tables
- feature importances

Currently not saved:

- fitted RF object
- fitted XGBoost object
- fitted lower depth model
- reusable ensemble artifact
- preprocessing/schema object for inference

This means the current attempt is excellent for analysis, but it is not yet a
deployable inference package.

Before evaluating a new external dataset as a true ready-model test, we need to
create a final export step that:

- refits the chosen model family on the original training data only
- saves the fitted estimator objects
- saves the feature schema
- saves the blend weights
- provides a script that loads those artifacts and predicts on new images

The external dataset must not be used for training or model selection.

## 16. Final Conclusions

The main conclusions from `attempt4` are:

- The best current model family is an RF+XGBoost ensemble.
- The best individual RF model is `raw_depth_geometry_metadata_rf`.
- The best individual XGBoost model is `raw_depth_geometry_metadata_xgb`.
- Raw relative depth improves both RF and XGBoost full models.
- Learned stacked depth features did not beat the simpler raw depth feature in this setup.
- Depth-only models are much weaker than geometry-based models.
- Bbox geometry remains the dominant signal.
- The model gets harder to trust as distance increases.
- The far range has a meaningful underprediction bias.
- The exact RF/XGBoost blend weight is not very sensitive around 50/50.
- The pipeline still needs final model export before real external inference.

## 17. Recommended Next Step

The next engineering step should be:

- create a final model export study for the RF+XGBoost ensemble
- save fitted RF and XGBoost models with `joblib`
- save preprocessing metadata and feature order
- then run a separate external evaluation on the prepared Nenrus dataset

That keeps the methodology clean:

- train only on our original dataset
- freeze the model
- test once on the external dataset
- report external MAE by drone type and distance

