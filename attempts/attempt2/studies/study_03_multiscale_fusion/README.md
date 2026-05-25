# Study 03: Multiscale Fusion

## Goal

Study 03 asks a new question:

- Is there one single best zoom level?
- Or is a multiscale combination better than any one zoom alone?

The earlier studies already showed that:

- `bbox_only__bbox_midpoint__object_depth` is the best single-scale method so far
- `bbox_expand_1_5x__bbox_midpoint__object_minus_ring` is the strongest context-aware alternative

This creates a natural next hypothesis:

- the tight crop may provide the cleanest drone-local signal
- medium and wide crops may provide useful scene-relative context
- combining them may outperform either one alone

## Why This Study Exists

The previous studies did not truly test whether context is useful in combination.
They only tested whether each context window could stand on its own after being
reduced to a single score.

That means the previous winner does **not** prove that context is useless.
It only proves that, in a single-scale single-score setup, the tight crop was
the strongest standalone representation.

Study 03 is meant to test the richer version of the theory:

- small zooms isolate the drone
- larger zooms keep more surrounding context
- a multiscale model may use both

## Recommended Context Windows

Core contexts:

- `bbox_only`
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- `bbox_expand_4x`

Additional wide-context ablations:

- `bbox_expand_10x`
- `bbox_expand_16x`
- `bbox_expand_6x`
- `bbox_expand_8x`
- `full_image`

Why these:

- `bbox_only` tests the cleanest drone-only signal
- `1.5x` and `2x` are the strongest balance candidates based on Study 02
- `4x` adds broader local scene structure
- `6x`, `8x`, `10x`, `16x`, and `full_image` test whether still more context
  adds anything beyond the proven local windows

## What We Will Compare

Stage A: extend the cached feature table with the wider zoom list

- reuse the exact Study 02 selected-image list
- carry over all existing Study 02 rows that already match the desired setup
- compute only the missing zoom windows: `6x`, `8x`, `10x`, and `16x`
- confirm whether any new single zoom beats the current best baseline

Stage B: multiscale fusion

- build one row per image from `representation_records.csv`
- keep separate per-scale features instead of collapsing each image to one score
- compare multiscale models against the best single-scale baselines

## Baselines To Beat

- `bbox_only__bbox_midpoint__object_depth`
- `bbox_expand_1_5x__bbox_midpoint__object_minus_ring`

If a multiscale model does not beat those, then the extra complexity is not yet
worth it.

## Planned Feature Families

Per-scale features:

- `object_depth`
- `object_depth_percentile_5_95`
- `object_minus_ring`
- `object_minus_ring_normalized`

Candidate cross-scale features:

- difference between tight and medium zoom scores
- difference between medium and wide zoom scores
- difference between local raw depth and local contrast scores

## Planned Models

Start simple:

- single-feature linear baselines
- multifeature linear regression
- ridge regression

Only move to more complex models if simple fusion clearly helps first.

## What Success Would Mean

The main success condition is:

- lower `LOOCV MAE` than the current best single-scale method

Useful secondary signs:

- lower RMSE
- lower relative error
- reduced near/far compression
- more stable performance across distance bins

## Notes

- This folder already contains a valid extraction/study config for the added
  zoom windows.
- `run_extend_feature_cache.py` builds a new Study 03 feature CSV by reusing the
  Study 02 sample list and only computing missing contexts.
- The current single-scale study scripts can still be reused after Stage A to
  rank all individual zoom windows inside the new Study 03 feature cache.
- Stage B will need a new fusion script that pivots `representation_records.csv`
  into one row per image and evaluates combined models.
