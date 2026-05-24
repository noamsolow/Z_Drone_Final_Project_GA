# Attempt 3: Stacked Depth, Big Random Forests, and Noisy-BBox Robustness

This document is the full recap of `attempt3`.

It explains:

- why this attempt was necessary after `attempt1` and `attempt2`
- what each study in `attempt3` was testing
- how the code and evaluation pipeline were built
- what the results were
- why we think the results look the way they do
- what the main graphs show
- what questions are still open

Attempt 3 became a three-step bridge between the earlier phases:

- `attempt1` gave us the original metric-distance calibration baselines
- `attempt2` gave us better depth-only feature engineering and the small depth-only random forest
- `attempt3` tested how those learned depth signals behave when plugged back into full models

In short:

- Study 01 asked whether the small RF depth feature improves the old linear calibration setup
- Study 02 asked whether a bigger RF on top changes that answer
- Study 03 asked whether the answer changes again once bbox geometry is made more realistic

## 1. Why Attempt 3 Exists

`attempt1` showed that a calibration model can predict metric distance using:

- one raw depth scalar
- bbox geometry
- scene metadata

But that attempt also left an important ambiguity:

- was the raw depth signal itself good, or was the model mostly riding on geometry?

Then `attempt2` deliberately zoomed in on the depth side of the problem.

It asked:

- what is the best way to represent relative depth around the drone?
- can a depth-only model learn a better signal than any single handcrafted depth statistic?

The answer from `attempt2` was yes:

- the small depth-only random forest was meaningfully better than any single raw depth feature

That naturally created the next question:

- if we take that learned depth signal and plug it back into the full metric-distance model, does the final system improve?

That question is exactly what `attempt3` was built to answer.

## 2. High-Level Structure

`attempt3` contains three studies:

- [Study 01](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01)
- [Study 02](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02)
- [Study 03](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03)

They form a clean progression.

### Study 01

Keep the large model linear, keep geometry and metadata unchanged, and replace only:

- old raw depth scalar

with:

- the output of the small depth-only RF

This was the purest apples-to-apples feature replacement test.

### Study 02

Keep the same stage-one depth idea, but replace the top linear model with a larger RF.

This tested whether the small RF still helps once the top model itself can learn non-linear interactions.

### Study 03

Keep the big-RF framework, but stop trusting an unrealistically perfect bbox.

Instead:

- jitter the bbox with small shifts and scale changes
- recompute geometry and depth summaries under those perturbations
- aggregate them into robust per-image features

This was the realism/robustness study.

## 3. Datasets and Feature Sources

Attempt 3 reused earlier artifacts whenever possible, but only reused expensive cached features, not fitted models.

### What we reused

From `attempt1`:

- [features.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt1/artifacts/features/features.csv)

From `attempt2` Study 06:

- [depth_only_feature_table.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/fused_features/depth_only_feature_table.csv)
- [subset_random_forest_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_metrics.csv)

### What we did not reuse blindly

We did **not** just take saved predictions from earlier studies and plug them in as-is.

Instead:

- the lower RF was refit inside the new split structure
- the top model was refit inside the new split structure

That matters because it prevents leakage.

### Split-safe principle

Whenever the upper model used the output of the lower model:

- that lower-model prediction was generated from a fit that did **not** train on the same validation/test row

This was one of the most important methodological rules of `attempt3`.

## 4. Core Evaluation Philosophy

Across the attempt, we cared about two things:

1. fairness
2. realism

### Fairness

When we changed one thing, we tried to keep the rest fixed.

Examples:

- Study 01 changed the depth feature but kept the top model linear
- Study 02 changed the top model family but kept the overall feature comparison structure

### Realism

Study 03 acknowledged that exact geometry from perfect labels is too optimistic.

So instead of pretending perfect boxes are realistic, it built a more robust representation around likely detector error.

## 5. Study 01: Linear Stacked Replacement

Main files:

- [run_study_01.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/run_study_01.py)
- [analyze_study_01.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/analyze_study_01.py)
- [study01 README](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/README.md)

### Goal

Study 01 asked:

- if we keep the old `attempt1` full model structure,
- and only replace the raw depth scalar with the small-RF depth output,
- do we improve the final metric-distance model?

### Data used

Study 01 had to use the intersection between:

- `attempt1` rows
- `attempt2` Study 06 rows

because they were not built on exactly the same image snapshot.

Observed counts:

- attempt1 rows: `14,800`
- Study 06 rows: `15,064`
- shared usable rows: `13,864`

That intersection was the cleanest fair comparison set available.

### Models compared

- `raw_depth_only_linear`
- `geometry_metadata_linear`
- `raw_depth_geometry_metadata_linear`
- `small_rf_depth_only_linear`
- `stacked_rf_depth_geometry_metadata_linear`

### Main result

Winner:

- `stacked_rf_depth_geometry_metadata_linear`

From [study01 summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/artifacts/reports/summary.json):

- CV OOF MAE: `12.78m`
- test MAE: `12.55m`

Key comparison:

- old full linear baseline: `13.58m`
- stacked linear model: `12.55m`

So Study 01 proved:

- the small RF did create a better depth feature than the old raw scalar
- and that better depth feature improved the final linear calibration model

### Why that mattered

This was the missing bridge between `attempt2` and the full calibration setup.

It showed that the small RF was not only a good depth-only model.  
It was also a useful feature generator for a downstream metric-distance estimator.

## 6. Study 02: Big RF on Top

Main files:

- [run_study_02.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/run_study_02.py)
- [study02 README](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/README.md)

### Goal

Study 02 asked:

- what happens if the top model is also a random forest?
- does the small RF depth feature still help once the final model itself is already non-linear?

### Models compared

- `raw_depth_only_rf`
- `geometry_metadata_rf`
- `raw_depth_geometry_metadata_rf`
- `small_rf_depth_only_rf`
- `stacked_rf_depth_geometry_metadata_rf`

### Candidate-depth sweep

We did not assume tree depth in advance.

For the upper RF, we evaluated:

- `rf_shallow`
- `rf_medium`
- `rf_deep`
- `rf_unbounded`

This was explicitly done to test:

- where deeper trees help
- where they begin to overfit

### Main result

Winner:

- `raw_depth_geometry_metadata_rf`
- candidate: `rf_unbounded`

From [study02 summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/artifacts/reports/summary.json):

- CV OOF MAE: `7.63m`
- test MAE: `7.46m`
- test mean relative error: `9.38%`

### What that meant

This was a very important result.

It showed that:

- the big RF was a huge improvement over the linear top model
- but the stacked small-RF depth feature did **not** win once the top model was also a strong RF

In other words:

- the small RF helped depth-only modeling
- the small RF helped the linear top model
- but with a powerful upper RF, raw depth + geometry + metadata slightly beat the stacked version

### Why we think that happened

The most likely reason is:

- the upper RF can already learn non-linear corrections from raw depth once geometry is present

So the small RF still creates a better depth-only signal, but its advantage shrinks once the upper model is strong enough to learn those corrections itself.

### Deeper lesson

Study 02 also made it clear that geometry was extremely powerful.

That raised the next concern:

- are the geometry results partly inflated by the fact that the bbox is ground-truth perfect?

That question led directly to Study 03.

## 7. Study 03: Noisy-BBox Aggregation

Main files:

- [run_study_03.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/run_study_03.py)
- [analyze_study_03.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/analyze_study_03.py)
- [make_noisy_bbox_examples.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/make_noisy_bbox_examples.py)
- [study03 README](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/README.md)

### Goal

Study 03 asked:

- if geometry is made less perfect and more detector-like,
- does the stacked depth feature become more valuable?

### What changed

Instead of one exact bbox per image, we created seven versions:

- original
- shift left
- shift right
- shift up
- shift down
- scale down
- scale up

Then we recomputed:

- geometry features for each bbox
- depth summaries for each bbox inside each context crop

And aggregated them into:

- `...__jitter_median`
- `...__jitter_std`

### Contexts and depth aggregations

Contexts:

- `1x`
- `1.5x`
- `2x`
- `4x`

Depth aggregations:

- `bbox_midpoint`
- `bbox_mean`
- `inner50_median`

### Why this was more realistic

In a real system, detections are not perfect.

Boxes can be:

- slightly shifted
- slightly too big or too small
- slightly inconsistent between frames

The noisy-bbox aggregation was designed to test robustness under that reality.

### Main result

Winner:

- `raw_depth_geometry_metadata_rf`
- candidate: `rf_unbounded`

From [study03 summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/summary.json):

- CV OOF MAE: `5.35m`
- test MAE: `5.21m`
- test mean relative error: `7.43%`

### Why this result was surprising

The big surprise was:

- performance improved a lot relative to Study 02, even though we made the bbox setup more realistic

That does **not** mean “noise is good” by itself.

What likely happened is:

- the median across jittered boxes created a more robust central estimate
- the standard deviation across jittered boxes created a new stability/reliability signal
- the model could use both

So this was not just degrading the box.

It was turning one exact measurement into a richer robust representation.

### Did the stacked model finally win?

No.

The stacked model remained very strong:

- test MAE: `5.33m`

But the winner was still:

- `raw_depth_geometry_metadata_rf`
- test MAE: `5.21m`

So even under more realistic bbox noise:

- the stacked model still did not beat raw depth + geometry + metadata

### What that means

This is actually an important validation result.

It tells us that the Study 02 conclusion was **not** only an artifact of perfect boxes.

Even after removing exact oracle geometry:

- robust geometry stayed dominant
- the big RF stayed the best top model
- raw depth + geometry + metadata still slightly beat the stacked version

## 8. Full Attempt Summary

Here is the whole attempt in one table.

| Study | Main Question | Winner | Test MAE |
| --- | --- | --- | ---: |
| Study 01 | Does the small RF help the old linear setup? | `stacked_rf_depth_geometry_metadata_linear` | `12.55` |
| Study 02 | Does the answer change with a big RF on top? | `raw_depth_geometry_metadata_rf` | `7.46` |
| Study 03 | Does the answer change again with realistic noisy bbox aggregation? | `raw_depth_geometry_metadata_rf` | `5.21` |

This progression matters.

It says:

1. the small RF is useful
2. but the top-model family matters even more
3. and robust aggregated geometry is extremely powerful

## 9. What We Think the Results Mean

This is our current best interpretation.

### Theory 1: The small RF really does improve depth-only signal

This part is well supported.

Across the attempt:

- raw depth alone is weak
- small-RF depth alone is much better

So the lower model is learning something real and useful.

### Theory 2: Geometry is still the dominant metric-distance signal

This is the biggest repeated pattern.

Even in Study 03, after adding bbox noise:

- geometry-based features still had the highest importances
- geometry-only remained very strong

So the system is not primarily a “depth-only” solution.

It is a:

- geometry-first
- depth-assisted

solution.

### Theory 3: The upper RF can already learn most of the useful raw-depth corrections

This is the likely explanation for why the stacked model stopped winning once the upper model became non-linear.

In Study 01:

- linear top model benefited from the cleaner small-RF depth feature

In Studies 02 and 03:

- the big RF likely learned enough correction directly from raw depth plus geometry

So the extra stage no longer produced the best final combination.

### Theory 4: Noisy aggregation helped by making features robust, not by making them worse

Study 03 improved performance because it gave the model:

- robust central measurements
- explicit instability measurements

The `__jitter_std` features were especially informative.

That means the model learned not only:

- “how big is the drone?”

but also:

- “how stable is that apparent size under small detector perturbations?”

That is a stronger, more realistic feature set.

## 10. Why the Results Improved So Much in Study 03

This is one of the main questions people naturally ask.

Why would “noisy bbox aggregation” outperform exact bbox features?

The most likely reasons are:

### Robust summary effect

One exact measurement can be brittle.

Using:

- jitter median
- jitter std

turns each feature into something more stable and informative.

### New reliability information

The model now knows whether a feature changes a lot under tiny bbox changes.

That is effectively a confidence-like signal.

### More complete data usage

Study 03 used all `15,064` images directly, whereas Study 02 used the smaller shared subset.

So not all of the improvement should be attributed only to the noisy-bbox design.

### Better inductive bias

The feature construction in Study 03 encourages the model to learn signals that survive realistic detection noise, which may generalize better than signals that only work for one perfect box.

## 11. Overfitting: What We Checked and What We Still Do Not Know

The strongest evidence against severe overfitting is:

- CV and held-out test stayed close

For example, in Study 03 winner:

- CV OOF MAE: `5.35`
- test MAE: `5.21`

That is a very good sign.

We also used:

- fixed holdout test
- outer CV
- leakage-safe lower-model fitting

So the results are not just train-set artifacts.

But there are still open generalization questions:

- how much is this tuned to this dataset distribution?
- how well would it transfer to truly new drone images?
- would it still hold up under detector outputs that are worse than our synthetic jitter patterns?

So the remaining concern is less:

- “classic memorization overfitting”

and more:

- “dataset-specific fit”

## 12. What Graphs Exist and How to Read Them

### Study 01 analysis package

Folder:

- [study01 analysis](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/artifacts/reports/analysis)

Useful graphs:

- [exact_distance_mae_all_models.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/artifacts/reports/analysis/exact_distance_mae_all_models.png)
  - shows how each linear model behaves across true distance
- [overall_model_comparison.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/artifacts/reports/analysis/overall_model_comparison.png)
  - quick overall winner comparison
- [stacked_improvement_mae_by_distance.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/artifacts/reports/analysis/stacked_improvement_mae_by_distance.png)
  - shows where stacking helped most

### Study 02 core result files

Folder:

- [study02 reports](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/artifacts/reports)

Useful files:

- [summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/artifacts/reports/summary.json)
- [test_metrics.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/artifacts/reports/test_metrics.json)
- [upper_rf_candidate_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/artifacts/reports/upper_rf_candidate_metrics.csv)
- [upper_rf_feature_importances.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/artifacts/reports/upper_rf_feature_importances.csv)

These are most useful for understanding:

- why the big RF won
- which tree-depth regime worked best
- how dominant geometry still was

### Study 03 analysis package

Folder:

- [study03 analysis](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis)

Best Study 03-only graphs:

- [study03_overall_model_comparison.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_overall_model_comparison.png)
  - compares all Study 03 models overall
- [study03_exact_distance_mae_all_models.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_exact_distance_mae_all_models.png)
  - shows exact-distance MAE curves
- [study03_exact_distance_relative_error_all_models.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_exact_distance_relative_error_all_models.png)
  - same idea, but relative error
- [study03_exact_distance_signed_error_all_models.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_exact_distance_signed_error_all_models.png)
  - shows overprediction/underprediction trends
- [study03_distance_range_comparison.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_distance_range_comparison.png)
  - near/mid/far comparison for the strongest models
- [study03_candidate_sweep.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_candidate_sweep.png)
  - shows which upper-tree-depth regime worked best
- [study03_winner_feature_importances.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study03_winner_feature_importances.png)
  - shows what the winning Study 03 model actually relied on

Best Study 02 vs Study 03 comparison graphs:

- [study02_vs_study03_winner_overall.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study02_vs_study03_winner_overall.png)
  - quickest overall winner comparison
- [study02_vs_study03_matched_model_families.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study02_vs_study03_matched_model_families.png)
  - compares the same model families across the two studies
- [study02_vs_study03_rawgeom_mae_by_distance.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study02_vs_study03_rawgeom_mae_by_distance.png)
  - shows how the Study 02 winner family changed under noisy aggregation
- [study02_vs_study03_rawgeom_relative_error_by_distance.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study02_vs_study03_rawgeom_relative_error_by_distance.png)
  - same comparison in relative-error space
- [study02_vs_study03_stacked_mae_by_distance.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study02_vs_study03_stacked_mae_by_distance.png)
  - shows whether the stacked model gained or lost after noisy aggregation
- [study02_vs_study03_winner_feature_importances.png](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/reports/analysis/study02_vs_study03_winner_feature_importances.png)
  - compares what the winning models relied on in both studies

### Noisy-bbox example images

Folder:

- [study03 examples](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/artifacts/examples)

These show:

- the original bbox
- the jittered bboxes
- the depth map for one selected context
- the local bbox placements inside that crop

They are useful for visually understanding what “noisy bbox aggregation” really means.

## 13. Questions We Still Have

Even after the strong Study 03 result, several good research questions remain.

### Question 1

Would the stacked model become the winner if geometry became even less reliable?

Study 03 only used small perturbations.

Maybe with:

- worse detector noise
- asymmetric boxes
- missed centering

the stacked depth feature would matter more.

### Question 2

Would a different top model beat RF?

We have not yet tested:

- XGBoost
- ExtraTrees
- boosting-style ensembles

It is possible that a different upper model would use the stacked depth signal better than RF did.

### Question 3

How much of the current success is still dataset-specific?

The strongest remaining generalization question is:

- does this still work as well on truly unseen data?

### Question 4

Can we make the geometry representation itself even more realistic?

Study 03 already made geometry better and more realistic, but it is still label-derived geometry under controlled perturbation, not detector-produced geometry.

### Question 5

Would temporal information help?

All current attempts are per-image.

If a future system used:

- frame-to-frame consistency
- motion cues
- temporal smoothing

the balance between depth and geometry might change again.

## 14. Code and Infrastructure Added in Attempt 3

This attempt also expanded the shared codebase a bit.

Shared helpers added or extended:

- [pipeline/calibration/linear_models.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/pipeline/calibration/linear_models.py)
- [pipeline/calibration/splits.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/pipeline/calibration/splits.py)

Study-level runner and analysis files:

- [study01 run](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/run_study_01.py)
- [study01 analysis](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study01/analyze_study_01.py)
- [study02 run](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study02/run_study_02.py)
- [study03 run](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/run_study_03.py)
- [study03 analysis](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/analyze_study_03.py)
- [study03 visualization helper](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/attempt3/studies/study03/make_noisy_bbox_examples.py)

## 15. Bottom Line

Attempt 3 answered the major questions it was meant to answer.

### What we established

- The small RF from `attempt2` really does learn a stronger depth-only signal than the original raw depth scalar.
- That stronger signal improves the old linear calibration setup.
- A big RF on top is much better than a linear top model.
- Once the top model is a strong RF, raw depth + geometry + metadata slightly outperforms the stacked depth feature.
- That conclusion still holds even after geometry is made more realistic with noisy bbox aggregation.
- Robust aggregated geometry is one of the strongest ideas discovered in the whole project so far.

### The best-performing configuration so far

Within `attempt3`, the strongest result is:

- Study 03 winner: `raw_depth_geometry_metadata_rf`
- test MAE: `5.21m`
- test mean relative error: `7.43%`

### The cleanest interpretation

The project is no longer pointing toward a pure depth solution.

It is pointing toward:

- a strong geometry-based metric-distance estimator
- improved by depth
- stabilized by robust aggregation
- and best modeled by a non-linear top model

That is the main story of `attempt3`.
