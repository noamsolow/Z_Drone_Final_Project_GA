# Full Attempt and Study Report

This version is expanded so it includes the MAEs that were explicitly reported
in the main study summaries, result tables, and model-comparison artifacts.

Metric note:

- `test MAE` means held-out test on the study split.
- `CV MAE`, `OOF MAE`, or `LOO MAE` means the study did not use one final held-out test in that stage.
- `real external MAE` means evaluation on the Nenrus real-drone dataset.

## Project Story in One Line

- Attempt 1 proved the full calibration idea works.
- Attempt 2 found the best depth representation and depth-only lower model.
- Attempt 3 found the best synthetic full model with robust noisy-bbox features.
- Attempt 4 turned that into a harder detector-like benchmark and found RF + XGBoost blending helps.
- Attempt 5 showed the synthetic model fails on real data unless calibrated.
- Attempt 6 tried to fix the real-domain problem without real training, but calibration still remained necessary.

## Attempt 1

### What was added in short

- First reusable full calibration pipeline.
- Used relative depth, bbox geometry, and metadata.
- Added deterministic train/dev/test splits and evaluation code.

### MAEs

#### Cross-validation ranking

| Model | CV MAE |
| --- | ---: |
| `depth_geometry_metadata_linear` | `14.63` |
| `depth_only_linear` | `32.20` |
| `scale_only` | `45.96` |

#### Held-out test

| Model | Test MAE |
| --- | ---: |
| `depth_geometry_metadata_linear` | `14.50` |
| `depth_only_linear` | `32.11` |
| `scale_only` | `47.75` |

### Explanation

The first real success was that the full model worked at all. The strongest
signal was bbox geometry, while raw depth alone was weak and a single scale
factor was not good enough.

## Attempt 2

### Study 01 - Pilot representation study

#### Added in short

- First systematic sweep over depth representations.
- Compared crop size, aggregation method, and score type.

#### MAEs

| Candidate | LOO MAE |
| --- | ---: |
| `bbox_only__bbox_midpoint__object_depth` | `25.17` |
| `bbox_only__bbox_midpoint__object_depth_percentile_5_95` | `26.54` |
| `bbox_expand_1_5x__bbox_midpoint__object_minus_ring` | `26.76` |

#### Explanation

The best single depth cue came from a very tight crop around the drone and the
midpoint depth value. Local object-centered depth beat wide-scene context.

### Study 02 - Expanded representation study

#### Added in short

- Reran the same idea on a much larger balanced sample.

#### MAEs

| Candidate | Main MAE |
| --- | ---: |
| `bbox_only__bbox_midpoint__object_depth` | `25.46` |
| `bbox_expand_1_5x__bbox_midpoint__object_minus_ring` | `26.27` |
| `bbox_only__bbox_midpoint__object_depth_percentile_5_95` | `26.82` |

#### Explanation

The same winner survived the larger rerun, which made the result much more
trustworthy. Tight local depth stayed best.

### Study 03 - Multiscale fusion

#### Added in short

- Added more zoom levels and tested fusion across depth features.

#### MAEs

| Model / Feature | MAE |
| --- | ---: |
| Best single feature `bbox_only__bbox_midpoint__object_depth` | `25.46` |
| Best fusion model `greedy_forward_ols_k6` | `24.41` |

#### Explanation

Fusion helped, but only a little. The useful fusion was local multiscale
fusion such as `bbox_only` plus `2x`, not very wide context.

### Study 04 - Depth-only lower model

#### Added in short

- First learned depth-only model stage.
- Compared single features, linear fusion, and tree models.

#### MAEs

| Model | CV MAE |
| --- | ---: |
| Best single feature `bbox_only__bbox_midpoint__object_depth` | `25.04` |
| Best linear model `greedy_forward_ols_k8` | `24.10` |
| Best overall model `random_forest_top_24` | `22.95` |

#### Explanation

This was the first strong proof that a learned depth-only lower model beats
both handcrafted single features and linear fusion.

### Study 05 - Relative depth tables

#### Added in short

- Diagnostic study that generated raw depth tables by distance and zoom.

#### MAEs

- No model training in this study.
- No MAE result.

#### Explanation

This was a support study used to understand the depth values, not to optimize
the final predictor.

### Study 06 - All-images selected-subset random forest

#### Added in short

- Moved from a balanced subset to the full dataset.
- Simplified the feature space to a smaller raw-depth-only subset.

#### MAEs

| Model | CV MAE |
| --- | ---: |
| Best single feature `bbox_only__bbox_midpoint__object_depth` | `24.21` |
| `random_forest_top_12` | `21.14` |
| `random_forest_top_24` | `20.92` |

#### Explanation

This confirmed that the depth-only lower-model idea from Study 04 was real and
not just a small-sample artifact.

## Attempt 3

### Study 01 - Linear stacked replacement

#### Added in short

- Replaced raw depth in the old full linear model with the learned lower-model
  depth signal from Attempt 2.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| `raw_depth_only_linear` | `31.06` |
| `geometry_metadata_linear` | `13.85` |
| `raw_depth_geometry_metadata_linear` | `13.58` |
| `small_rf_depth_only_linear` | `20.90` |
| `stacked_rf_depth_geometry_metadata_linear` | `12.55` |

#### Explanation

The learned depth-only lower model really helped once it was used as a feature
inside the full model. It improved the old linear setup.

### Study 02 - Big random forest on top

#### Added in short

- Replaced the upper linear model with a larger RF.
- Compared raw depth against stacked learned depth in the same full pipeline.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| `raw_depth_only_rf` | `30.75` |
| `geometry_metadata_rf` | `7.69` |
| `raw_depth_geometry_metadata_rf` | `7.46` |
| `small_rf_depth_only_rf` | `21.27` |
| `stacked_rf_depth_geometry_metadata_rf` | `7.54` |

#### Explanation

A strong non-linear top model mattered more than the stacked lower model. Once
the upper model became powerful, raw depth plus geometry plus metadata slightly
beat the stacked version.

### Study 03 - Noisy-bbox aggregation

#### Added in short

- Jittered the bounding boxes.
- Recomputed depth and geometry under those jitters.
- Aggregated robust median and variability features.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| `raw_depth_only_rf` | `23.28` |
| `geometry_metadata_rf` | `5.40` |
| `raw_depth_geometry_metadata_rf` | `5.21` |
| `small_rf_depth_only_rf` | `19.06` |
| `stacked_rf_depth_geometry_metadata_rf` | `5.33` |

#### Explanation

This was the best synthetic full-model result in the project. The gain came
from robust geometry and jitter-stability features, not from noise alone.

## Attempt 4

### Study 01 - RF on expanded noisy rows

#### Added in short

- Switched from one robust row per image to several noisy bbox rows per image.
- This is harder and more detector-like.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| `raw_depth_only_rf` | `26.7110` |
| `geometry_metadata_rf` | `8.0335` |
| `raw_depth_geometry_metadata_rf` | `7.7124` |
| `small_rf_depth_only_rf` | `20.6553` |
| `stacked_rf_depth_geometry_metadata_rf` | `8.0300` |

#### Explanation

The task got harder, but geometry still dominated and raw depth still helped as
an extra correction signal.

### Study 02 - XGBoost on the same expanded rows

#### Added in short

- Kept the same data as Study 01.
- Changed only the model family from RF to XGBoost.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| `raw_depth_only_xgb` | `26.6562` |
| `geometry_metadata_xgb` | `8.1890` |
| `raw_depth_geometry_metadata_xgb` | `7.7301` |
| `small_xgb_depth_only_xgb` | `20.9286` |
| `stacked_xgb_depth_geometry_metadata_xgb` | `7.9000` |

#### Explanation

XGBoost told the same story as RF: depth-only was weak, geometry was strong,
and raw depth helped a little but did not become the main signal.

### Study 03 - RF + XGBoost blend

#### Added in short

- Blended the best RF and XGBoost predictions.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| `rf_winner_baseline` | `7.7124` |
| `xgb_winner_baseline` | `7.7301` |
| `equal_weight_blend` | `7.6325` |
| `tuned_weight_blend` | `7.6347` |

#### Explanation

The blend improved slightly over either single model. The gain was small but
real enough to make the RF + XGBoost ensemble the best family in this setup.

## Attempt 5

### Study 01 - Exported final ensemble

#### Added in short

- Saved RF, XGBoost, ensemble metadata, and feature schema for reuse.

#### MAEs

| Model | Test MAE |
| --- | ---: |
| Tuned blend export | `7.6347` |
| Equal blend export | `7.6325` |

#### Explanation

This study turned the best Attempt 4 result into a reusable saved artifact and
reproduced the known synthetic performance.

### Study 02 - External Nenrus evaluation

#### Added in short

- Ran the frozen exported synthetic model on Nenrus real-drone data.

#### MAEs

| Evaluation slice | Real external MAE |
| --- | ---: |
| Overall Nenrus | `26.1078` |
| `Kongsberg` | `36.1536` |
| `Vestfold` | `18.9774` |

#### Explanation

The frozen synthetic model failed badly on real data. The dominant problem was
geometry domain shift: real drones had different bbox scale behavior.

### Study 03 - Depth-weighted external variants

#### Added in short

- Tested whether making depth more influential reduces the Nenrus failure.

#### MAEs

| Variant | Ensemble MAE | RF MAE | XGB MAE |
| --- | ---: | ---: | ---: |
| `baseline_saved_ensemble` | `26.1078` | - | - |
| `depth_plus_normalized_geometry` | `21.6107` | `21.8472` | `21.4273` |
| `depth_repeated_16_with_geometry` | `26.4260` | `29.1235` | `24.3065` |
| `depth_repeated_8_with_geometry` | `26.7349` | `28.6084` | `25.2627` |
| `depth_plus_no_size_geometry` | `42.7872` | `43.6192` | `42.1445` |
| `depth_only` | `43.2253` | `42.8687` | `43.9051` |

#### Explanation

Depth helped only when paired with some normalized geometry. Pure depth was not
strong enough, and simply repeating the depth feature did not fix the domain
shift.

### Study 04 - Calibration on top of frozen model

#### Added in short

- Learned a small calibration function on top of the frozen ensemble.
- Did not retrain RF or XGBoost.

#### Honest split-evaluation MAEs

##### 10% calibration data

| Calibration model | MAE |
| --- | ---: |
| `raw_uncalibrated` | `26.2094` |
| `global_scale_only` | `7.7856` |
| `global_affine` | `7.7721` |
| `global_quadratic` | `7.6001` |
| `per_drone_affine` | `3.4435` |
| `per_drone_quadratic` | `3.1614` |

##### 20% calibration data

| Calibration model | MAE |
| --- | ---: |
| `raw_uncalibrated` | `26.1586` |
| `global_scale_only` | `7.7287` |
| `global_affine` | `7.7144` |
| `global_quadratic` | `7.5341` |
| `per_drone_affine` | `3.4065` |
| `per_drone_quadratic` | `3.0735` |

##### 30% calibration data

| Calibration model | MAE |
| --- | ---: |
| `raw_uncalibrated` | `26.1701` |
| `global_scale_only` | `7.7412` |
| `global_affine` | `7.7282` |
| `global_quadratic` | `7.5461` |
| `per_drone_affine` | `3.3720` |
| `per_drone_quadratic` | `3.0529` |

##### 50% calibration data

| Calibration model | MAE |
| --- | ---: |
| `raw_uncalibrated` | `26.2176` |
| `global_scale_only` | `7.7763` |
| `global_affine` | `7.7648` |
| `global_quadratic` | `7.5742` |
| `per_drone_affine` | `3.3662` |
| `per_drone_quadratic` | `3.0499` |

#### Final full-data calibration fit

| Model | MAE |
| --- | ---: |
| Raw uncalibrated model | `26.1078` |
| Final per-drone quadratic calibration | `3.0251` |

#### Explanation

This was the key real-world result. The frozen synthetic model alone did not
generalize, but a small per-drone calibration layer corrected the scale error
and made the system usable on real Nenrus data.

## Attempt 6

### Study 01 - Geometry-regularized synthetic-to-real generalization

#### Added in short

- Tried geometry dropout, geometry noise, reduced geometry, and depth-only
  variants.
- Still trained only on synthetic rows.

#### Synthetic holdout MAEs

| Variant | Synthetic holdout MAE |
| --- | ---: |
| `baseline_full_geometry` | `7.6347` |
| `geometry_dropout_030` | `8.1062` |
| `reduced_geometry` | `8.4325` |
| `geometry_dropout_060` | `8.6675` |
| `geometry_noise_020` | `8.7748` |
| `geometry_dropout_030_noise_020` | `9.1324` |
| `raw_depth_only` | `26.7251` |

#### Real external MAEs

| Variant | Real external MAE |
| --- | ---: |
| `reduced_geometry` | `21.5652` |
| `baseline_full_geometry` | `26.3380` |
| `geometry_dropout_060` | `28.9492` |
| `geometry_noise_020` | `29.3322` |
| `geometry_dropout_030` | `30.2391` |
| `geometry_dropout_030_noise_020` | `30.4439` |
| `raw_depth_only` | `40.1370` |

#### Explanation

Reducing geometry helped the real-domain result the most, but none of these
synthetic-only regularization tricks solved the generalization problem.

### Study 02 - BBox scale domain randomization

#### Added in short

- Added generic synthetic bbox-scale randomization.
- Still did not use real rows for fitting or scale selection.

#### Synthetic holdout MAEs

| Variant | Synthetic holdout MAE |
| --- | ---: |
| `baseline_full_geometry` | `7.6347` |
| `reduced_geometry` | `8.4325` |
| `scale_uniform_060_140_full_geometry` | `9.2134` |
| `scale_uniform_040_140_full_geometry` | `10.4177` |
| `scale_lognormal_full_geometry` | `10.6579` |
| `scale_uniform_040_140_reduced_geometry` | `10.9250` |
| `scale_lognormal_reduced_geometry` | `11.1483` |
| `raw_depth_only` | `26.7251` |

#### Real external MAEs

| Variant | Real external MAE |
| --- | ---: |
| `reduced_geometry` | `21.5652` |
| `scale_uniform_040_140_reduced_geometry` | `22.5309` |
| `scale_uniform_040_140_full_geometry` | `25.8499` |
| `baseline_full_geometry` | `26.3380` |
| `scale_lognormal_reduced_geometry` | `27.3601` |
| `scale_uniform_060_140_full_geometry` | `29.8799` |
| `scale_lognormal_full_geometry` | `31.6300` |
| `raw_depth_only` | `40.1370` |

#### Explanation

Scale randomization did not beat the simpler reduced-geometry setup. The
remaining real-domain problem is bigger than random bbox scaling alone.

## Final Conclusions

1. The strongest synthetic full-model result was Attempt 3 Study 03 with test
   MAE `5.21m`.
2. The strongest harder detector-like synthetic result was Attempt 4 Study 03
   with equal-weight blend test MAE `7.6325m`.
3. The strongest real-data result came from Attempt 5 Study 04 with honest
   held-out MAE `3.0735m` at 20% calibration data, and `3.0251m` after the
   final full-data fit.
4. Geometry is the most powerful synthetic cue, but it is also the main source
   of real-domain failure.
5. Relative depth helps, but mostly as a correction signal and not as the only
   predictor.
6. If the goal is deployment on real data, the best path so far is:
   - keep the frozen ensemble idea from Attempt 4 / Attempt 5
   - reduce geometry over-reliance where possible
   - calibrate on the target real domain
