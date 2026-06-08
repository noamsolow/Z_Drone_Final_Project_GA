# Attempt6 Study03 Analysis

## Setup

This ablation reused the cached Attempt4 Study01 noisy expanded feature table and copied the original grouped holdout/CV split files.
Only the raw depth feature feeding `raw_depth_geometry_metadata_rf` was changed.

## Results

- Baseline `bbox_only__inner50_median__object_depth`: CV OOF MAE = 7.790668, test MAE = 7.712390
- Comparison `bbox_only__bbox_midpoint__object_depth`: CV OOF MAE = 7.971710, test MAE = 7.948575
- CV delta (`bbox_midpoint - inner50_median`) = +0.181042
- Test delta (`bbox_midpoint - inner50_median`) = +0.236186

## Conclusion

For this exact historical RF rerun, `bbox_only__inner50_median__object_depth` finished better on held-out test MAE than `bbox_only__bbox_midpoint__object_depth`.
The historical `inner50_median` choice remains the safer option for this exact RF setup. A full rerun of later studies is not justified by this ablation.
