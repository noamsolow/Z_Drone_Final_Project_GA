# BBox Domain Shift and Simple Calibration Analysis

## Geometry Comparison

The original dataset uses only the `jitter_name=original` rows from Attempt 4, so the comparison is between real labelled boxes rather than augmented scale jitter rows.

Shared distances between Original and Nenrus are `20, 30, 40, 50, 60, 70`.

### bbox_width_norm: Real / Original Mean Ratio

- `Kongsberg` `20`m: real/original ratio `0.4437`, real mean `0.0173`, original mean `0.0389`
- `Kongsberg` `30`m: real/original ratio `0.4584`, real mean `0.0124`, original mean `0.0272`
- `Kongsberg` `40`m: real/original ratio `0.5503`, real mean `0.0103`, original mean `0.0187`
- `Kongsberg` `50`m: real/original ratio `0.5960`, real mean `0.0086`, original mean `0.0144`
- `Vestfold` `20`m: real/original ratio `0.7586`, real mean `0.0295`, original mean `0.0389`
- `Vestfold` `30`m: real/original ratio `0.6834`, real mean `0.0186`, original mean `0.0272`
- `Vestfold` `40`m: real/original ratio `0.8945`, real mean `0.0167`, original mean `0.0187`
- `Vestfold` `50`m: real/original ratio `0.8785`, real mean `0.0126`, original mean `0.0144`
- `Vestfold` `60`m: real/original ratio `0.9478`, real mean `0.0120`, original mean `0.0127`
- `Vestfold` `70`m: real/original ratio `0.6899`, real mean `0.0080`, original mean `0.0116`

### bbox_area_ratio: Real / Original Mean Ratio

- `Kongsberg` `20`m: real/original ratio `0.1191`, real mean `0.0002`, original mean `0.0017`
- `Kongsberg` `30`m: real/original ratio `0.1406`, real mean `0.0001`, original mean `0.0008`
- `Kongsberg` `40`m: real/original ratio `0.2180`, real mean `0.0001`, original mean `0.0004`
- `Kongsberg` `50`m: real/original ratio `0.2592`, real mean `0.0001`, original mean `0.0002`
- `Vestfold` `20`m: real/original ratio `0.3937`, real mean `0.0007`, original mean `0.0017`
- `Vestfold` `30`m: real/original ratio `0.3810`, real mean `0.0003`, original mean `0.0008`
- `Vestfold` `40`m: real/original ratio `0.6837`, real mean `0.0003`, original mean `0.0004`
- `Vestfold` `50`m: real/original ratio `0.7435`, real mean `0.0002`, original mean `0.0002`
- `Vestfold` `60`m: real/original ratio `0.7717`, real mean `0.0001`, original mean `0.0002`
- `Vestfold` `70`m: real/original ratio `0.4361`, real mean `0.0001`, original mean `0.0001`

## Simple Calibration on 20% of Nenrus

- `per_drone_affine`: MAE `3.4065` +/- `0.0698`, mean relative error `0.1426`, within 10m `0.9443`
- `global_affine`: MAE `7.7144` +/- `0.0663`, mean relative error `0.2981`, within 10m `0.7009`
- `global_scale_only`: MAE `7.7287` +/- `0.0765`, mean relative error `0.3044`, within 10m `0.6988`
- `raw_uncalibrated`: MAE `26.1586` +/- `0.1428`, mean relative error `0.9995`, within 10m `0.0746`

## Interpretation

The geometry comparison directly tests whether the model sees real-drone boxes as larger or smaller than same-distance original-dataset boxes.
If real/original ratios are below 1.0, the real boxes are smaller than the original boxes at the same distance; since the model heavily relies on bbox size, that tends to push predictions upward.

The calibration experiment is not a pure external test anymore, because it uses a small labelled subset of Nenrus.
It answers a different question: if we are allowed a small calibration set from the real domain, can a simple mapping from predicted distance to true distance fix the systematic bias?
