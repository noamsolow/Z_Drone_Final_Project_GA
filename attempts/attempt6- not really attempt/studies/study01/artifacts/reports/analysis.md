# Attempt6 Study01 Analysis

This study trains only on synthetic rows and uses the real Nenrus rows as an external test.

## Winner on Real External Blend MAE

- Variant: `reduced_geometry`
- Real MAE: `21.5652m`
- Real mean relative error: `0.8662`
- Real overprediction rate: `0.9980`

## Important Note

The real dataset is not used to fit any RF, XGB, categorical levels, dropout medians, or noise parameters.