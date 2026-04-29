# First Attempt Calibration Study

This document summarizes the first full calibration-analysis attempt for the
project. It is meant to be readable by someone new to the repository and to
explain not only what we ran, but also why we chose these steps and what we
learned from the results.

## 1. Goal

The goal of this first attempt was to take the existing monocular depth
pipeline and convert it into a repeatable calibration study that predicts the
drone's true metric distance `true_distance_m` from:

- relative depth estimated by the monocular depth model
- bounding-box geometry from the YOLO label
- dataset metadata encoded in the filename (`weather`, `time_of_day`)

This study was designed to stay aligned with the midterm report in the
following ways:

- synthetic-data-first calibration stage
- explicit conversion from relative depth to metric distance
- comparison against a single global scale baseline
- evaluation with `MAE`, `Relative Error`, and error by distance range
- use of scene conditions such as weather and time of day

At the same time, this first attempt intentionally deferred a few midterm ideas
to later iterations:

- no real-world domain-gap validation yet
- no detector confidence feature yet
- no polynomial regression yet
- no per-range calibration model yet

## 2. Dataset Used

The study used the synthetic dataset loaded by `pipeline.data.loader`.

Observed dataset summary:

- Total records used: `14,800`
- Weather conditions:
  - `clear_sky`: `7,344`
  - `light_rain`: `7,456`
- Time-of-day conditions:
  - `10AM`: `9,734`
  - `8PM`: `5,066`
- Unique ground-truth distances:
  - `20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 125, 150` meters

The metadata distribution came directly from the cached feature summary in
`first_attempt/artifacts/features/summary.json`.

## 3. What We Built

We added a reusable first-attempt study on top of the existing pipeline.

### Reusable calibration package code

Under `pipeline/calibration/` we added:

- `features.py`
  - builds flat regression-ready rows from the loader and depth pipeline
- `splits.py`
  - creates deterministic held-out and cross-validation splits
- `linear_models.py`
  - implements the three first-attempt baselines
- `evaluation.py`
  - computes metrics, grouped error summaries, and feature correlations

### Study workflow code

Under `first_attempt/` we added:

- `config.yaml`
- `run_extract_features.py`
- `run_study.py`

These scripts create the cached feature table and then run the modeling study
from that cache.

## 4. Workflow

The workflow had two stages.

### Stage A: Feature extraction

Command:

```bash
python first_attempt/run_extract_features.py
```

This stage:

1. Iterates over the dataset with `DroneDatasetLoader(strict=True)`.
2. Loads each RGB image.
3. Runs the current depth model on the full image.
4. Extracts one representative depth value for the drone.
5. Computes geometry and metadata features.
6. Saves the result to:
   - `first_attempt/artifacts/features/features.csv`
   - `first_attempt/artifacts/features/summary.json`

### Stage B: Study and evaluation

Command:

```bash
python first_attempt/run_study.py
```

This stage:

1. Loads `features.csv`.
2. Creates or reloads deterministic splits.
3. Runs 5-fold cross-validation on the development set.
4. Compares the three candidate models.
5. Chooses the winner by mean CV `MAE`, with mean CV relative error as
   the tie-break.
6. Refits each model on the full development set.
7. Evaluates all models once on the held-out test set.
8. Writes result files under `first_attempt/artifacts/reports/`.

## 5. Train / Validation / Test Strategy

The midterm report proposed a `70 / 15 / 15` split with preserved
representation across distance ranges and weather conditions.

For this implementation, we kept the same held-out test philosophy but made one
practical improvement:

- `15%` held-out test set
- `85%` development set
- `5-fold` cross-validation inside the development set instead of one fixed
  validation split

Why we did this:

- it preserves a clean unseen test set
- it gives a more stable model comparison than one validation split
- it reduces the chance that one lucky validation split chooses the wrong model

Actual split sizes:

- Test set: `2,220` samples
- Development set: `12,580` samples
- CV fold validation sizes: about `2,512` to `2,519` samples each
- CV fold training sizes: about `10,061` to `10,068` samples each

Stratification was done on the combined label:

- distance range:
  - `near = 20 to 60 m`
  - `mid = 70 to 100 m`
  - `far = 115 to 150 m`
- `weather`
- `time_of_day`

This was chosen so that each split preserves not just distance diversity but
also scene-condition diversity.

## 6. Feature Engineering

Each row in `features.csv` contains:

- target:
  - `true_distance_m`
- depth feature:
  - `relative_depth`
- geometry features:
  - `bbox_width_px`
  - `bbox_height_px`
  - `bbox_width_norm`
  - `bbox_height_norm`
  - `bbox_area_ratio`
  - `bbox_aspect_ratio`
  - `bbox_center_x_norm`
  - `bbox_center_y_norm`
- metadata:
  - `weather`
  - `time_of_day`
- identifiers:
  - `image_path`
  - `label_path`

### Why these features were chosen

`relative_depth`
- This is the most direct signal from the depth model.
- Even though it is not metric, it is the natural starting point for
  calibration.

Bounding-box width / height / area
- Perspective theory says farther drones occupy fewer pixels.
- The midterm report also emphasized using geometric cues from the detected
  region and mentioned size-related parametrization ideas.

Normalized geometry
- Pixel dimensions alone depend on image resolution.
- Normalized terms are easier to compare across runs and safer for future data.

Bounding-box center
- We included center position to check whether vertical or horizontal image
  location changes model behavior.
- For example, sky-heavy vs horizon-heavy regions might bias depth estimates.

Weather and time of day
- The dataset explicitly controls scene conditions.
- The midterm report highlighted the value of controlled synthetic conditions
  and the importance of testing the effect of such variations.

## 7. Why We Used the Median Depth, Not the Mean

The drone-relative depth value is extracted by:

```python
relative_depth = np.median(depth_map[y1:y2, x1:x2])
```

We chose the median, not the arithmetic mean, for several reasons:

- the bounding box contains both drone pixels and background pixels
- monocular depth maps can have noisy outliers inside a small region
- the mean is easily pulled by a few extreme pixels
- the median is more robust when the drone occupies only part of the box

Why not use only the center pixel:

- a single pixel is too fragile
- it may land on a box edge, the sky, or a noisy region in the depth map

Why not use the average:

- if part of the bbox contains distant background, the average can drift away
  from the drone signal

So the median is a practical compromise:

- more stable than one pixel
- more robust than the mean
- easy to compute

## 8. Models We Compared

The midterm report proposed a gradual calibration development process. Our
three first-attempt models follow that philosophy.

### Model 1: `scale_only`

Formula:

```text
Z ≈ s * d_rel
```

with:

```text
s = sum(d_rel * Z_true) / sum(d_rel^2)
```

Why we chose it:

- this is the simplest and most classic calibration baseline
- it directly matches the current existing scale-calibration code
- it answers the question: "Is one global conversion factor enough?"

### Model 2: `depth_only_linear`

Formula:

```text
Z ≈ b0 + b1 * d_rel
```

Why we chose it:

- it is the smallest upgrade over Model 1
- it tests whether adding an intercept already helps
- it answers the question: "Does relative depth contain useful information even
  if one global scale factor is too simple?"

### Model 3: `depth_geometry_metadata_linear`

Inputs:

- `relative_depth`
- all bbox geometry features
- one-hot encoding for `weather`
- one-hot encoding for `time_of_day`

Why we chose it:

- it tests the main first-attempt hypothesis:
  - raw depth alone is not enough
  - geometry and scene metadata may explain part of the remaining error
- it stays interpretable
- it is still linear, so we can understand feature influence more easily
- it creates a solid bridge toward later polynomial or piecewise models

## 9. Preprocessing Choices

### One-hot encoding for metadata

We encoded:

- `weather=light_rain`
- `time_of_day=8PM`

Why:

- linear regression needs numeric inputs
- one-hot encoding is simple and interpretable

### Standardization of continuous features

Continuous features were standardized using training-fold statistics only.

Why:

- this prevents data leakage
- it makes coefficient estimation numerically more stable
- it keeps CV fair

The model coefficients are then exported back in the original feature names for
readability.

## 10. Evaluation Metrics

The evaluation follows the midterm report closely.

### MAE

`MAE` = mean absolute error in meters.

Why:

- this is the most intuitive physical-space metric
- it tells us how many meters off we are on average

### Mean Relative Error

Formula:

```text
abs(predicted - true) / true
```

Why:

- the same meter error means different things at different distances
- a 2-meter error at 10 meters is much more serious than a 2-meter error at
  100 meters

### RMSE

Why:

- this punishes large errors more strongly than MAE
- useful for spotting whether a model makes occasional severe failures

### R²

Why:

- this summarizes how much of the distance variation the model explains
- useful at the overall level

### Error by distance range

Why:

- the midterm explicitly warned that global averages can hide failures at
  certain distances
- this is especially important when the drone becomes very small in the image

## 11. What the Raw Feature Correlations Showed

From `correlations.csv`:

- `relative_depth`: `0.035`
- `bbox_width_px`: `-0.762`
- `bbox_height_px`: `-0.629`
- `bbox_width_norm`: `-0.758`
- `bbox_height_norm`: `-0.630`
- `bbox_area_ratio`: `-0.535`

Interpretation:

- raw relative depth alone had almost no simple linear correlation with
  ground-truth distance
- bbox size features had much stronger correlation
- bigger boxes usually mean a closer drone, which is exactly what perspective
  suggests

This is important because it strongly supports one of the key midterm concerns:
relative depth is inconsistent across images because monocular depth models
normalize each scene differently. That means raw `d_rel` does not carry a
stable metric meaning by itself.

## 12. Final Results

### Cross-validation ranking

From `cv_metrics.json`:

1. `depth_geometry_metadata_linear`
   - mean CV `MAE = 14.63 m`
   - mean CV relative error `= 28.38%`
2. `depth_only_linear`
   - mean CV `MAE = 32.20 m`
   - mean CV relative error `= 65.72%`
3. `scale_only`
   - mean CV `MAE = 45.96 m`
   - mean CV relative error `= 70.51%`

### Held-out test results

From `test_metrics.json`:

| Model | MAE (m) | Mean Relative Error | RMSE (m) | R² |
| --- | ---: | ---: | ---: | ---: |
| `scale_only` | 47.75 | 75.11% | 58.19 | -1.33 |
| `depth_only_linear` | 32.11 | 65.57% | 38.14 | -0.00 |
| `depth_geometry_metadata_linear` | 14.50 | 27.79% | 18.29 | 0.77 |

Main takeaway:

- the best model was clearly `depth_geometry_metadata_linear`
- the single global scale model performed very poorly
- the depth-only linear model improved on that baseline, but remained weak
- the richer linear model produced a large improvement and was the only model
  that looked genuinely useful

## 13. Distance-Range Results

For the winning model:

| Range | Count | MAE (m) | Mean Relative Error |
| --- | ---: | ---: | ---: |
| Near | 913 | 16.01 | 49.69% |
| Mid | 746 | 9.37 | 11.45% |
| Far | 561 | 18.90 | 13.88% |

Interpretation:

- the model worked best in the `mid` range
- `far` was not as accurate in absolute meters, but was still much better than
  the simpler baselines
- `near` was the weakest regime by relative error

This means the first attempt succeeded overall, but it is not equally strong at
all distances.

## 14. What Worked Best and Worst

### What worked best

- combining depth with geometry and metadata
- using deterministic splits and cross-validation
- evaluating separately by distance range instead of relying only on one global
  metric

### What worked worst

- assuming one global scale factor is enough
- relying on relative depth alone
- expecting the same calibration behavior across all distance regimes

## 15. Why We Think the Results Look This Way

### Why the full model won

The model likely won because:

- bbox size is strongly related to distance in this dataset
- metadata helps explain scene-specific biases
- relative depth still adds some signal after geometry is included

In other words, the model is not just learning one depth-to-meters conversion.
It is also learning corrective context.

### Why near distances were hardest

Likely reasons:

- the linear model tends to regress toward a global middle range
- in the test summary, near predictions were biased upward:
  - mean true near distance: about `40.26 m`
  - mean predicted near distance: about `46.73 m`
- this creates a large relative error even when the absolute meter error is not
  the worst in the whole dataset

Possible interpretation:

- the model is overcompensating and pulling close drones too far outward

### Why mid distances were best

Likely reasons:

- this is the regime where both geometry and depth cues are still informative
- the drone is not too large and not too tiny
- the model's global linear assumptions fit this region best

### Why far distances were better than near in relative terms

Likely reasons:

- even though far-range MAE is higher in meters, the denominator is also larger
- relative error is therefore smaller
- geometry features may still preserve a decent monotonic trend at long range

### Why `light_rain` and `8PM` were slightly better than `clear_sky` and `10AM`

From `grouped_error_summary.csv`, the winning model had slightly lower error in:

- `light_rain` than `clear_sky`
- `8PM` than `10AM`

We do not treat this as a universal physical conclusion yet.
It may reflect:

- dataset-specific rendering differences
- better drone/background contrast in some conditions
- accidental correlations with drone scale appearance

This should be tested further before drawing a strong conclusion.

## 16. Important Limitations

### Runtime was not instrumented

This first attempt did not record wall-clock runtime per stage.

That means:

- we know the pipeline completed successfully
- but we do not yet have trustworthy timing numbers for:
  - feature extraction
  - per-image depth inference
  - study runtime

Future runs should explicitly log:

- total extraction time
- average time per sample
- total study time
- average fit/evaluation time per model

### Coefficients should be interpreted carefully

Some exported linear coefficients are numerically large.
This does not automatically mean those features are individually dominant.

Why:

- geometry features are correlated with each other
- normalized and pixel-space versions overlap in meaning
- linear regression can distribute weight across correlated inputs in unstable
  ways

So for interpretation:

- trust error metrics first
- use coefficients only as rough directional hints

### This is still synthetic-only

The midterm report also emphasized the future domain-gap stage.
This first attempt says nothing yet about real-world generalization.

## 17. How We Can Improve Next

These are the main improvement ideas suggested by the first attempt.

### 1. Compare median vs mean vs center pixel

Theory:

- median may be helping, but it may not be optimal for every distance regime

How to test:

- rerun feature extraction with:
  - median inside bbox
  - mean inside bbox
  - center pixel only
  - trimmed mean
- compare CV and test metrics

### 2. Use a tighter drone region than the full bounding box

Theory:

- bbox background leakage is still hurting the depth signal, especially at far
  range

How to test:

- shrink the box before extracting depth
- use only the central 50% or 60% of the bbox
- compare against the full-box median

### 3. Add per-range or piecewise calibration

Theory:

- one global linear rule is too coarse
- near, mid, and far may need different calibration behavior

How to test:

- fit one model per range
- or fit a piecewise model with range-dependent coefficients
- compare to the current global model on held-out test

This idea is directly consistent with the midterm's proposed next step after a
weak baseline.

### 4. Add polynomial terms

Theory:

- the relation between geometry and metric distance is not fully linear

How to test:

- add squared terms such as:
  - `bbox_width_norm^2`
  - `bbox_height_norm^2`
  - `relative_depth^2`
- compare against the current linear winner

This also matches the midterm roadmap.

### 5. Add detector confidence

Theory:

- low-confidence detections may correspond to unreliable geometry

How to test:

- once real detector outputs are included, add confidence as a feature
- compare grouped error for high- and low-confidence cases

This idea was explicitly mentioned in the midterm discussion.

### 6. Reduce feature redundancy

Theory:

- current geometry features overlap heavily
- this may make the linear fit harder to interpret and possibly less stable

How to test:

- run ablations:
  - depth only
  - depth + width/height only
  - depth + normalized geometry only
  - depth + metadata only
- compare CV and test metrics

### 7. Add robust or regularized regression

Theory:

- a few hard samples may still be distorting the fit

How to test:

- compare current OLS against:
  - ridge regression
  - lasso
  - robust regression

### 8. Add real-world evaluation

Theory:

- synthetic improvements may not fully survive the domain gap

How to test:

- collect a small real-image evaluation set with known distances
- report the same metrics there
- compare synthetic vs real grouped errors

## 18. How to Check Whether Our Theories Are Real

The best way to avoid storytelling without evidence is to turn each theory into
an experiment.

Recommended checks:

- Residual plots:
  - plot `prediction - truth` vs `true_distance_m`
  - confirms whether near is overestimated and far is underestimated
- Feature ablation:
  - remove one feature group at a time
  - shows whether geometry, metadata, or depth is doing most of the work
- Alternative depth aggregation:
  - compare median vs mean vs center crop
- Per-range calibration:
  - compare one global model against near/mid/far-specific models
- Condition-specific analysis:
  - compare errors by `weather`, `time_of_day`, and their combinations
- Repeated-split stability:
  - try multiple seeds for the held-out split
  - confirms whether the result is stable or split-dependent

## 19. Files Produced by This Attempt

### Feature artifacts

- `first_attempt/artifacts/features/features.csv`
- `first_attempt/artifacts/features/summary.json`

### Split artifacts

- `first_attempt/artifacts/splits/test_split.json`
- `first_attempt/artifacts/splits/cv_folds.json`

### Report artifacts

- `first_attempt/artifacts/reports/cv_metrics.json`
- `first_attempt/artifacts/reports/test_metrics.json`
- `first_attempt/artifacts/reports/correlations.csv`
- `first_attempt/artifacts/reports/grouped_error_summary.csv`
- `first_attempt/artifacts/reports/model_coefficients.csv`
- `first_attempt/artifacts/reports/test_predictions.csv`

## 20. Bottom Line

This first attempt was successful as a calibration study.

What we learned:

- a single global scale factor is not enough
- relative depth alone is not enough
- bbox geometry is highly informative
- adding metadata improves the model further
- the best first-attempt model is the linear model that combines depth,
  geometry, and metadata
- the main unresolved weakness is near-range performance

The next most logical steps are:

- test better depth aggregation inside the bbox
- add per-range or polynomial calibration
- reduce background leakage
- add detector confidence
- validate on real-world data

That path is fully consistent with the direction laid out in the midterm
report, while being grounded in actual measurements from this first complete
experiment.
