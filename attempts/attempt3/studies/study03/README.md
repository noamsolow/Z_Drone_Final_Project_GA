# Study 03: Noisy-BBox Aggregation Random Forest

This study extends `attempt3/study02` toward a more realistic detector setting.

## Motivation

In Study 02, geometry was extremely strong because the drone bounding box came
from the ground-truth label and was therefore unrealistically precise.

That creates an important gap:

- the model may look excellent because bbox size and position are effectively oracle inputs
- in a real deployment, detected boxes will be slightly shifted, slightly scaled, and not perfectly stable

Study 03 addresses that by replacing exact bbox-driven features with features
aggregated over several small bbox perturbations.

## What Changes Relative to Study 02

The high-level modeling protocol stays the same:

- use all images
- use a small depth-only random forest as the lower model
- compare upper random-forest models with and without the lower-model output
- evaluate with fixed holdout + stratified CV

The key change is feature construction.

Instead of trusting one exact bbox, the study:

1. builds several small shifted/scaled versions of the bbox
2. recomputes geometry from those jittered boxes
3. recomputes depth summaries from those jittered boxes inside each context crop
4. aggregates the resulting values into one robust feature row per image

## Depth Feature Setup

Contexts:

- `bbox_only` (`1x`)
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- `bbox_expand_4x`

Aggregation methods:

- `bbox_midpoint`
- `bbox_mean`
- `inner50_median`

For each context/method pair, the study stores:

- `...__jitter_median`
- `...__jitter_std`

The lower small-RF uses the `__jitter_median` depth features as its input
feature set.

## Geometry Feature Setup

For each jittered bbox, the study computes the same geometry family used in the
earlier attempts:

- width / height in pixels
- width / height normalized
- area ratio
- aspect ratio
- center x / y normalized

It then aggregates those across jittered boxes into:

- `...__jitter_median`
- `...__jitter_std`

This keeps the geometry informative, but removes the unrealistic exactness of a
single perfect box.

## Jitter Pattern

The default config uses seven boxes per image:

- original
- left shift
- right shift
- up shift
- down shift
- slightly smaller scale
- slightly larger scale

These are intentionally small perturbations, meant to approximate detector
imperfections rather than gross failure.

## Models Compared

- `raw_depth_only_rf`
- `geometry_metadata_rf`
- `raw_depth_geometry_metadata_rf`
- `small_rf_depth_only_rf`
- `stacked_rf_depth_geometry_metadata_rf`

The raw-depth baseline uses:

- `bbox_only__inner50_median__object_depth__jitter_median`

The stacked model replaces that raw depth scalar with:

- `depth_lower_model_prediction`

produced by the noisy-feature small RF.

## Leakage Handling

The study keeps the same leakage-safe design as Study 02:

- the lower random forest is cross-fit inside each outer split
- the upper model never receives a lower-model prediction from a fit that saw
  the same validation/test row

## Runtime Design

The expensive part is the noisy depth-feature build.

To make it practical, the study:

- writes one fused per-image feature row at a time
- flushes to disk immediately
- can resume from the existing feature table
- only computes missing images on rerun

## Main Outputs

Feature artifacts:

- `artifacts/features/noisy_aggregated_feature_table.csv`
- `artifacts/features/summary.json`
- `artifacts/features/selected_samples.csv`

Evaluation artifacts:

- `artifacts/reports/cv_metrics.json`
- `artifacts/reports/test_metrics.json`
- `artifacts/reports/cv_predictions.csv`
- `artifacts/reports/test_predictions.csv`
- `artifacts/reports/grouped_error_summary.csv`
- `artifacts/reports/upper_rf_candidate_metrics.csv`
- `artifacts/reports/upper_rf_selected_models.csv`
- `artifacts/reports/upper_rf_feature_importances.csv`
- `artifacts/reports/lower_model_feature_importances.csv`
- `artifacts/reports/summary.json`

## Run

Dry run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt3/studies/study03/run_study_03.py" --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt3/studies/study03/run_study_03.py"
```

## Expected Research Value

If Study 02 overestimated the power of geometry because the bbox was too exact,
Study 03 should answer that directly.

The main thing to watch is:

- whether the stacked model becomes more competitive once geometry is made less perfect

If that happens, it would support the idea that the small RF becomes more
valuable in more realistic detector conditions.
