# Second Attempt: Relative Depth Representation and Depth-Only Modeling Study

This document summarizes the full workflow of the second attempt. It is
written as a continuation of the first attempt README: attempt 1 asked how far
we can get with a direct calibration model, while attempt 2 asks a more
focused question first:

How should we represent relative depth around the drone before we build the
next full model?

This phase grew into four studies because the answer was not a single yes/no
result. We first had to understand the best single depth representation, then
check whether that result was stable on more images, then test multiscale
fusion, and finally test whether a learned depth-only lower model can extract
a stronger signal from the same relative-depth features.

The goal of this README is to explain:

- what motivated each study
- what we built and ran
- what each method means in simple terms
- what the results were
- why the results look the way they do
- what we now conclude and what should happen next

## 1. Goal

The first attempt showed two things at the same time:

- a full calibration model using depth, geometry, and metadata could work
- the raw depth feature by itself was not strong enough

That created a natural next question:

Before we combine depth with anything else, what is the best way to extract
the drone's depth signal from the image?

So the second attempt intentionally separated the problem into stages.

Instead of immediately building another full final model, we first studied:

- what crop should go into the depth model
- how the drone region should be summarized
- what final scalar depth score should be used
- whether one zoom is enough, or multiple zooms should be combined
- whether a learned depth-only model can outperform a single hand-designed
  score

## 2. How This Continues Attempt 1

Attempt 1 used the full image, extracted one representative depth value from
the drone box, and mixed that with geometry and metadata inside a regression
pipeline.

Attempt 2 deliberately sits between attempt 1 and the future final model:

- it keeps the same dataset and same base depth model
- it removes geometry and metadata from the main question at first
- it studies the depth representation itself as the object of interest

In other words:

- attempt 1 asked: "Can a calibration model work at all?"
- attempt 2 asks: "What is the strongest relative-depth signal we can give
  that future model?"

That is why this phase is best read as a depth-representation and depth-only
modeling phase, not as a final production model phase.

## 3. Dataset Used

This phase uses the same synthetic dataset root as the earlier work:

- `C:\Users\depthlev\Desktop\droneImages\dataset`

The studies do not use the entire dataset directly. Instead, they select a
balanced sample over exact condition strata:

- `true_distance_m`
- `weather`
- `time_of_day`

Observed values:

- distances: `20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 125, 150`
- weather: `clear_sky`, `light_rain`
- time of day: `10AM`, `8PM`

That gives:

- `12` distance values
- `2` weather values
- `2` time values
- total `48` exact strata

Why this matters:

- every study stays balanced across both easy and hard distances
- weather and time are represented evenly
- comparisons between methods are less likely to be driven by one lucky subset

## 4. What We Built

The second attempt added a reusable study layer on top of the shared pipeline.

### Reusable pipeline code

Under `pipeline/` we rely on:

- `pipeline/depth/depth_model.py`
  - loads the relative-depth model
- `pipeline/depth/representation.py`
  - defines context windows, aggregation logic, and score fields
- `pipeline/calibration/depth_representation_features.py`
  - builds cached per-image representation records
- `pipeline/calibration/depth_representation_study.py`
  - evaluates single-feature candidate methods
- `pipeline/calibration/evaluation.py`
  - computes overall metrics and grouped summaries

### Study workflow code

Under `attempts/second attempt/` and its study folders we added:

- `run_extract_features.py`
- `run_study.py`
- `studies/study_03_multiscale_fusion/run_extend_feature_cache.py`
- `studies/study_03_multiscale_fusion/analyze_study_03.py`
- `studies/study_04_depth_only_models_30_per_stratum/run_extend_feature_cache.py`
- `studies/study_04_depth_only_models_30_per_stratum/run_depth_only_model_study.py`
- `studies/study_04_depth_only_models_30_per_stratum/analyze_study_04.py`

### Single-image sanity-check tools

We also added two single-image inspection folders:

- `test_single_image_depth_mask_baseline`
- `test_single_image_context_window_maps`

These are important because they let us visually inspect what the depth model
is doing before trusting the larger tables.

## 5. Overall Workflow

This attempt ended up having four main studies.

### Study 01: Pilot representation study

- small balanced sample
- goal: find the first promising representation family quickly

### Study 02: Expanded representation study

- same idea, much larger sample
- goal: check whether the pilot winner survives

### Study 03: Multiscale fusion study

- keep the same sampled images as Study 02
- add much wider zoom contexts
- goal: check whether multiple zooms together beat the best single zoom

### Study 04: Depth-only lower-model study

- larger balanced sample
- build a learned model from depth features only
- goal: find the strongest depth-only combination before building the final
  stacked model

### Shared execution pattern

Across the phase, the workflow usually followed one of these patterns.

#### Studies 01 and 02

1. build the cached representation table
2. rank all single-feature methods

Commands:

```bash
python "attempts/second attempt/run_extract_features.py" --config "<study-config>"
python "attempts/second attempt/run_study.py" --config "<study-config>"
```

#### Study 03

1. extend the cached feature table with additional zooms
2. rerun the single-feature ranking
3. run the multiscale fusion study
4. generate analysis graphs

Commands:

```bash
python "attempts/second attempt/studies/study_03_multiscale_fusion/run_extend_feature_cache.py"
python "attempts/second attempt/run_study.py" --config "attempts/second attempt/studies/study_03_multiscale_fusion/config.yaml"
python "attempts/second attempt/studies/study_03_multiscale_fusion/analyze_study_03.py"
```

#### Study 04

1. extend the cache to `30` images per stratum
2. run the single-feature ranking on the larger cache
3. run the depth-only lower-model study
4. generate graphs and summary analysis

Commands:

```bash
python "attempts/second attempt/studies/study_04_depth_only_models_30_per_stratum/run_extend_feature_cache.py"
python "attempts/second attempt/run_study.py" --config "attempts/second attempt/studies/study_04_depth_only_models_30_per_stratum/config.yaml"
python "attempts/second attempt/studies/study_04_depth_only_models_30_per_stratum/run_depth_only_model_study.py"
python "attempts/second attempt/studies/study_04_depth_only_models_30_per_stratum/analyze_study_04.py"
```

## 6. Why We Split Attempt 2 Into Multiple Studies

We did not split the work into multiple studies just for organization. Each
study answered a different scientific question.

### Study 01 asked:

- which single depth representation looks best at all?

### Study 02 asked:

- is that result stable when we use more images?

### Study 03 asked:

- can multiple zoom levels together beat the best single zoom?

### Study 04 asked:

- if we use only depth-derived features, can a learned model extract a
  stronger signal than any single hand-designed score?

This sequencing matters because each study built directly on the previous one
instead of mixing too many variables at once.

## 7. Representation Design Choices

The central object of this phase is a depth representation. Each candidate
method is defined by three choices:

1. the context window sent into the depth model
2. the aggregation rule used to summarize the drone region
3. the final score field used as the scalar feature

### 7.1 Context windows

The initial studies used these five context windows:

- `full_image`
- `bbox_expand_4x`
- `bbox_expand_2x`
- `bbox_expand_1_5x`
- `bbox_only`

Study 03 and Study 04 expanded the list to nine:

- `full_image`
- `bbox_expand_16x`
- `bbox_expand_10x`
- `bbox_expand_8x`
- `bbox_expand_6x`
- `bbox_expand_4x`
- `bbox_expand_2x`
- `bbox_expand_1_5x`
- `bbox_only`

What they mean:

- `bbox_only`
  - only the drone box is shown to the depth model
- `bbox_expand_1_5x` and `bbox_expand_2x`
  - the drone plus immediate local surroundings
- `bbox_expand_4x` and wider
  - progressively broader scene context
- `full_image`
  - the entire scene

Why we tested them:

- a tight crop might isolate the drone better
- a wider crop might give more scene-relative context
- the right answer was not obvious in advance

### 7.2 Aggregation methods

We tested three ways to summarize the drone region from the depth map:

- `inner50_median`
  - median depth in the central `50%` of the box
- `bbox_mean`
  - mean depth over the whole box
- `bbox_midpoint`
  - the center pixel depth

Why we tested them:

- `inner50_median` is robust and center-focused
- `bbox_mean` checks whether whole-box averaging helps
- `bbox_midpoint` checks whether the most central point is the cleanest cue

### 7.3 Score fields

For each `context window x aggregation method`, we evaluated four score
fields.

#### `object_depth`

- the raw object depth score

This asks:

- is the object's raw depth value already enough?

#### `object_depth_percentile_5_95`

- the raw score normalized into the crop's robust `5th-95th percentile` range

This asks:

- does crop-level normalization make the score more stable across images?

#### `object_minus_ring`

- object depth minus the surrounding-ring depth

This asks:

- is local contrast against nearby background more stable than raw depth?

#### `object_minus_ring_normalized`

- local object-vs-ring contrast, divided by the robust global depth range

This asks:

- should local contrast also be normalized for scale?

### 7.4 Important caveat for `bbox_only`

For `bbox_only`, the ring-based scores are structurally weak, because there is
no real surrounding area left inside the crop.

In those cases, the implementation falls back to:

- `surrounding_ring_depth = object_depth`

That forces:

- `object_minus_ring = 0`
- `object_minus_ring_normalized = 0`

So for `bbox_only`, the ring-difference variants should not be interpreted as
true competitors. They are degenerate by construction.

## 8. Evaluation Strategy and Metrics

### Studies 01 to 03

The first three studies evaluate single features or feature combinations
against the selected sample and ask a representation question.

The single-feature studies use leave-one-out calibration:

- fit a simple linear relationship on all images except one
- predict the held-out image
- repeat for every image

For a single feature, the model is:

```text
distance ~= intercept + slope * score
```

This keeps the evaluation simple and lets us compare representations fairly.

### Study 03 fusion model

Study 03 also tested a multifeature linear fusion model:

- `greedy_forward_ols_k6`

What that means:

- `greedy_forward`
  - start with no features, then add the next feature that improves error the
    most
- `ols`
  - ordinary least squares linear regression
- `k6`
  - keep six features total

### Study 04 lower-model study

Study 04 used a depth-only feature table and compared:

- single-feature baselines
- linear multifeature models
- tree-based models

The main model families were:

- `ols`
  - ordinary linear regression
- `ridge`
  - linear regression with coefficient shrinkage
- `random_forest`
  - many decision trees averaged together
- `extra_trees`
  - a more randomized tree ensemble
- `gradient_boosting`
  - sequential trees that correct earlier errors
- `xgboost`
  - a stronger boosted-tree implementation

Study 04 used balanced `5-fold` cross-validation and also added paired
bootstrap comparisons to estimate whether differences between top models were
meaningful rather than just lucky.

### Metrics used

Across the attempt, we tracked:

- `MAE`
  - average absolute distance error in meters
- `Median Absolute Error`
  - typical absolute error, less sensitive to outliers
- `Mean Relative Error`
  - average error relative to the true distance
- `RMSE`
  - penalizes large mistakes more strongly
- `R^2`
  - explained variance
- `Pearson`
  - linear relationship strength
- `Spearman`
  - ranking relationship strength
- `P90` and `P95` absolute error
  - tail-error summaries
- `within_5m`, `within_10m`, `within_20m`
  - practical hit-rate summaries

We also broke results down by:

- `near`, `mid`, `far`
- exact distance
- weather
- time of day
- weather x distance
- time x distance

## 9. Study 01: Pilot Representation Study

Folder:

- `studies/study_01_pilot_2_per_stratum`

### Motivation

We needed a fast first answer before investing in larger runs.

The pilot study asked:

- if we evaluate all reasonable single-feature representation choices, what
  family rises to the top?

### What we used

- `2` images per exact stratum
- `48` strata
- total `96` images
- total `1,440` representation rows
- `5` context windows
- `3` aggregation methods
- `4` score fields
- total `60` candidate methods

### What we did

For each selected image:

1. load the RGB image
2. generate each context crop
3. run the depth model on that crop
4. compute all three aggregation variants
5. compute all four score fields
6. evaluate each method with leave-one-out linear calibration

### Main result

Winner:

- `bbox_only__bbox_midpoint__object_depth`
- `MAE = 25.17 m`

Runner-up:

- `bbox_only__bbox_midpoint__object_depth_percentile_5_95`
- `MAE = 26.54 m`

Third:

- `bbox_expand_1_5x__bbox_midpoint__object_minus_ring`
- `MAE = 26.76 m`

### What we learned

The most surprising first result was:

- the best single method used only the drone crop
- the center pixel beat the mean and the inner median
- raw depth beat normalized depth in the winning method

This did not prove the result was fully stable, but it gave a clear starting
point for the larger rerun.

## 10. Study 02: Expanded Representation Study

Folder:

- `studies/study_02_expanded_10_per_stratum`

### Motivation

The pilot winner was plausible, but the gap at the top was not large enough to
trust blindly. Study 02 asked:

- does the pilot winner survive when the sample becomes much larger?

### What we used

- `10` images per exact stratum
- `48` strata
- total `480` images
- total `7,200` representation rows
- same `60` methods as Study 01

### Main result

Winner again:

- `bbox_only__bbox_midpoint__object_depth`
- `MAE = 25.46 m`
- `Mean Relative Error = 48.1%`
- `RMSE = 31.24 m`
- `R^2 = 0.337`

Runner-up:

- `bbox_expand_1_5x__bbox_midpoint__object_minus_ring`
- `MAE = 26.27 m`

Third:

- `bbox_only__bbox_midpoint__object_depth_percentile_5_95`
- `MAE = 26.82 m`

### What strengthened

- the same winner survived a `5x` larger image sample
- tight crops still dominated the top of the ranking
- `bbox_midpoint` remained the strongest aggregation family
- `full_image` remained clearly weak

### What weakened

- the gap between first and second place became smaller
- that means the exact winner remained stable, but not overwhelmingly dominant

### What this means

Study 02 gave us a more trustworthy conclusion than Study 01:

- the best standalone depth signal is a tight drone crop with `bbox_midpoint`
  and raw `object_depth`
- the strongest alternative is local context around `1.5x`, especially through
  `object_minus_ring`

That pointed directly to the next question:

- can we combine the strengths of tight and local-context crops?

## 11. Study 03: Multiscale Fusion Study

Folder:

- `studies/study_03_multiscale_fusion`

### Motivation

By the end of Study 02, we had two competing intuitions:

- `bbox_only` gave the strongest clean standalone signal
- `1.5x` local context was still very competitive

So Study 03 asked:

- is there a multiscale combination that beats any single zoom?

### What we used

- the same `480` selected images as Study 02
- `9` context windows
- `3` aggregation methods
- `4` score fields
- total `108` candidate depth features
- total `12,960` representation rows

### What we did

Study 03 had two layers.

#### Layer A: expanded single-feature ranking

We first reran the single-feature comparison with additional zooms:

- `6x`
- `8x`
- `10x`
- `16x`

This checked whether broader context might become better once we search a
wider zoom range.

#### Layer B: multiscale fusion

We then built one fused feature row per image and compared multifeature linear
fusion models, especially greedy forward OLS.

The core idea was simple:

- let the model choose whether it wants depth features from multiple zooms
  rather than forcing one zoom to win alone

### Main result

Best single feature still:

- `bbox_only__bbox_midpoint__object_depth`
- `MAE = 25.46 m`

Best fusion model:

- `greedy_forward_ols_k6`
- `MAE = 24.41 m`
- `RMSE = 30.11 m`
- `R^2 = 0.385`

Improvement vs best single:

- `MAE = -1.05 m`
- `RMSE = -1.14 m`
- `Mean Relative Error = -0.024`

### What the winning fusion model used

The six features were:

- `bbox_only__bbox_midpoint__object_depth`
- `bbox_only__inner50_median__object_depth_percentile_5_95`
- `bbox_only__inner50_median__object_depth`
- `bbox_only__bbox_midpoint__object_depth_percentile_5_95`
- `bbox_expand_2x__inner50_median__object_minus_ring`
- `bbox_expand_2x__bbox_midpoint__object_minus_ring`

### What this means

Study 03 did find a useful multiscale improvement, but not in the way we
originally imagined.

The winning multiscale model did not rely on:

- `6x`
- `8x`
- `10x`
- `16x`
- `full_image`

Instead, it combined:

- a strong `bbox_only` core
- a small amount of local-context correction from `2x`

So the study supports:

- local multiscale fusion

and does not support:

- broad scene-scale fusion as the main strategy

## 12. Study 04: Depth-Only Lower Model

Folder:

- `studies/study_04_depth_only_models_30_per_stratum`

### Motivation

Study 03 showed that combining multiple depth-derived features can help.

The next question was:

- can a learned depth-only model extract a stronger signal than any single
  hand-designed score or simple linear fusion?

Importantly, Study 04 was designed to stay depth-only.

That means:

- no bbox geometry as predictive input
- no weather as predictive input
- no time of day as predictive input

Those fields were still kept for grouped analysis, but not fed into the model.

### What we used

- `30` images per exact stratum
- `48` strata
- total `1,440` images
- total `38,880` representation rows
- `9` context windows
- `3` aggregation methods
- `4` score fields
- total `108` candidate depth-only features

### What we did

Study 04 had three layers.

#### Layer A: build the large depth-only cache

We reused compatible cached rows from earlier studies and computed only the
missing rows for the larger sample.

#### Layer B: compare model families

We built one fused depth-only feature row per image and compared:

- best single features
- OLS and Ridge multifeature models
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost

#### Layer C: check whether gains look meaningful

We added:

- balanced `5-fold` cross-validation
- paired bootstrap confidence intervals
- paired model comparisons between the top models

### Main result

Best single feature:

- `bbox_only__bbox_midpoint__object_depth`
- `CV MAE = 25.04 m`

Best linear multifeature model:

- `greedy_forward_ols_k8`
- `CV MAE = 24.10 m`

Best overall model:

- `random_forest_top_24`
- `CV MAE = 22.95 m`
- `CV RMSE = 28.69 m`
- `CV R^2 = 0.441`

Improvement vs best single:

- `MAE = -2.09 m`
- `RMSE = -2.40 m`
- `Mean Relative Error = -0.048`

### Significance summary

Paired bootstrap comparison:

- `best_model_vs_best_single`
  - mean MAE delta `= -2.09 m`
  - `95% CI [-2.67, -1.48]`

- `best_tree_vs_best_linear`
  - mean MAE delta `= -1.15 m`
  - `95% CI [-1.63, -0.68]`

This is important because the intervals stay below zero. That means the tree
model's gain over the best single feature and over the best linear model looks
real, not just like random ranking noise.

### What the best model actually used

The best model was still dominated by local depth families:

- `bbox_only`
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- with only a small supporting role from `bbox_expand_4x`

So even when we allow a stronger learned model, the study still points to the
same depth story:

- tight crop for the core signal
- immediate local context for correction

## 13. What the Graphs Show

The graph-heavy analysis appears mainly in Study 03 and Study 04.

### 13.1 Study 03 graph reading guide

Main folder:

- [Study 03 analysis](studies/study_03_multiscale_fusion/artifacts/reports/analysis)

#### [best_mae_by_context.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/best_mae_by_context.png)

What it shows:

- the best single method inside each zoom context

What it teaches:

- `bbox_only` is best overall
- `1.5x` is the strongest context-relative alternative
- very wide zooms are clearly weaker

#### [score_field_zoom_trends.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/score_field_zoom_trends.png)

What it shows:

- how each score-field family behaves as zoom changes

What it teaches:

- `object_depth` works best in very tight crops
- `object_minus_ring` works best in `1.5x` and `2x`
- wider zooms dilute all score families

#### [context_score_heatmap.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/context_score_heatmap.png)

What it shows:

- the best score field inside each context family

What it teaches:

- the best score is not the same at every zoom
- zoom and score field interact strongly

#### [single_vs_fusion_summary.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/single_vs_fusion_summary.png)

What it shows:

- best single feature vs best multiscale fusion model

What it teaches:

- multiscale fusion helps
- but the gain is modest, not huge

#### [distance_range_comparison.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/distance_range_comparison.png)

What it shows:

- best single vs best fusion across `near`, `mid`, and `far`

What it teaches:

- fusion helps more at the hard extremes
- the middle distance band was already relatively good

#### [exact_distance_zoom_trends.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/exact_distance_zoom_trends.png)

and

#### [best_context_winner_per_exact_distance.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/best_context_winner_per_exact_distance.png)

What they show:

- which zoom works best at each exact distance

What they teach:

- `bbox_only` wins most distances
- wider context only becomes competitive in the middle band
- even there, the winner is not consistent enough to replace tight crops

#### [weather_comparison.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/weather_comparison.png)

and

#### [time_of_day_comparison.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/time_of_day_comparison.png)

What they teach:

- weather and time matter somewhat
- but distance structure matters more

#### [weather_distance_mae_heatmaps.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/weather_distance_mae_heatmaps.png)

and

#### [time_distance_mae_heatmaps.png](studies/study_03_multiscale_fusion/artifacts/reports/analysis/time_distance_mae_heatmaps.png)

What they teach:

- the worst cells are still driven mainly by distance extremes
- condition effects are secondary

### 13.2 Study 04 graph reading guide

Main folder:

- [Study 04 analysis](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis)

#### [top_single_features.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/top_single_features.png)

What it shows:

- the strongest standalone depth features

What it teaches:

- tight and local-context features still dominate
- broad-scene depth is still not the main source of useful signal

#### [top_models.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/top_models.png)

What it shows:

- the full Study 04 model leaderboard

What it teaches:

- tree models dominate the top of the table
- this is not one lucky tree row; it is a family-level result

#### [best_single_linear_tree_summary.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/best_single_linear_tree_summary.png)

What it shows:

- best single feature vs best linear fusion vs best tree model

What it teaches:

- every step adds value
- but the tree model adds the most

#### [bootstrap_ci_mae.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/bootstrap_ci_mae.png)

What it shows:

- MAE confidence intervals for the top models

What it teaches:

- the best tree stays below the best single and best linear model in a way
  that supports a real gain

#### [best_single_by_context.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/best_single_by_context.png)

and

#### [context_score_heatmap.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/context_score_heatmap.png)

What they teach:

- the best single-context story did not change in Study 04
- `bbox_only`, `1.5x`, and `2x` still define the useful depth family

#### [best_tree_feature_importances.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/best_tree_feature_importances.png)

What it shows:

- which features the winning tree model relied on most

What it teaches:

- the tree still mainly trusts local relative-depth cues
- it does not suddenly shift to giant-scene cues

#### [distance_range_comparison.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/distance_range_comparison.png)

#### [exact_distance_mae_comparison.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/exact_distance_mae_comparison.png)

#### [weather_comparison.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/weather_comparison.png)

#### [time_of_day_comparison.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/time_of_day_comparison.png)

#### [weather_distance_heatmap_best_tree.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/weather_distance_heatmap_best_tree.png)

#### [time_distance_heatmap_best_tree.png](studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/analysis/time_distance_heatmap_best_tree.png)

Together these show:

- distance range is still the biggest driver of model difficulty
- the learned depth-only model improves broadly, but especially where the task
  is already hard
- weather and time remain secondary effects

## 14. What Worked Best and Worst

### What worked best

- `bbox_only__bbox_midpoint__object_depth` as the strongest single feature
- `bbox_expand_1_5x__bbox_midpoint__object_minus_ring` as the strongest
  context-relative alternative
- `bbox_expand_2x` as a useful correction scale in multiscale settings
- tree-based depth-only models once enough balanced data was available

### What worked poorly

- `full_image` as a standalone source of drone distance signal
- very wide zooms like `8x`, `10x`, `16x` as dominant solutions
- ring-based scores on `bbox_only`
- assuming one global simple linear score would solve the whole range

## 15. Why We Think the Results Look This Way

The overall pattern is actually quite consistent once we separate local and
global context.

### Why `bbox_only` keeps winning as a single feature

- the drone signal is least diluted there
- the center pixel often lands on the drone body rather than background
- broad scene content does not get a chance to dominate the depth-map scale

### Why `1.5x` and `2x` keep helping

- they preserve just enough nearby background to define a meaningful local
  contrast
- that makes `object_minus_ring` useful
- they add context without drowning the object

### Why broader scene contexts usually lose

- the drone becomes tiny relative to the crop
- the depth model must explain buildings, sky, horizon, and clutter
- when reduced to a small number of scalar features, that extra context adds
  more noise than help

### Why tree models helped in Study 04

- the best relationships are not fully linear
- different depth features matter in different image situations
- a tree ensemble can learn nonlinear combinations and conditional use of
  features without forcing one fixed slope for everyone

## 16. Important Limitations

- this is still synthetic-data-only analysis
- the depth model outputs relative depth, not metric depth directly
- some top gaps in the early studies were modest
- `bbox_only` ring-based variants are degenerate and should not be over-read
- even the best depth-only model still has meaningful error at the hardest
  distances
- Study 04 is a lower-model study, not the final stacked production model

## 17. What We Conclude and What Comes Next

The second attempt gives a much clearer answer than we had at the start.

### Main scientific conclusion

The strongest depth signal is local, not global.

More specifically:

- the best standalone depth feature is a very tight crop:
  - `bbox_only__bbox_midpoint__object_depth`
- the best supporting context comes from immediate surrounding regions:
  - mostly `1.5x` and `2x`
- broad-scene context does not become the main solution

### Main modeling conclusion

There is value in building a depth-only lower model before the final stacked
model.

Study 04 shows that:

- a learned depth-only model can beat the best handcrafted single score
- a tree model can beat the best linear fusion model
- the gain looks statistically meaningful, not random

### What should happen next

The next attempt should likely:

1. keep the Study 04 depth-only lower model idea
2. use its output as a strong learned depth feature
3. combine that lower-model output with the additional non-depth signals from
   attempt 1 in a final stacked model
4. test whether that full model improves near/far performance further

## 18. Files Produced by This Attempt

Main study folders:

- `studies/study_01_pilot_2_per_stratum`
- `studies/study_02_expanded_10_per_stratum`
- `studies/study_03_multiscale_fusion`
- `studies/study_04_depth_only_models_30_per_stratum`

Most important report files:

- Study 01 / 02
  - `artifacts/reports/study_summary.json`
  - `artifacts/reports/method_metrics.csv`
  - `artifacts/reports/grouped_error_summary.csv`
  - `artifacts/reports/loo_predictions.csv`

- Study 03
  - `artifacts/reports/fusion_study_summary.json`
  - `artifacts/reports/fusion_model_metrics.csv`
  - `artifacts/reports/analysis/study_03_analysis.md`

- Study 04
  - `artifacts/reports/depth_only_study_summary.json`
  - `artifacts/reports/depth_only_model_metrics.csv`
  - `artifacts/reports/depth_only_paired_model_comparisons.csv`
  - `artifacts/reports/analysis/study_04_analysis.md`

## 19. Bottom Line

Attempt 2 succeeded in answering the question it set out to study.

We started by asking how to represent the drone's relative depth better.

We ended with a stronger and more precise answer:

- the best single depth signal is a tight crop around the drone
- local context helps, but mainly at `1.5x` and `2x`
- broad-scene context is not the main path forward
- a learned depth-only model can improve significantly over any single
  handcrafted score

So the main output of attempt 2 is not just a winning score. It is a much
clearer depth-design strategy for the next full model.
