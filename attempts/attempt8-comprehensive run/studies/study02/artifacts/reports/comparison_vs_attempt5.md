# Study02 Comparison vs Attempt 5

Both studies use the same Nenrus rows and the same calibration protocol.

## Overall MAE

| Stage | Attempt 5 MAE | Attempt 8 MAE | Attempt 8 - Attempt 5 |
|---|---:|---:|---:|
| raw_uncalibrated_full_dataset | 26.1078m | 23.7959m | -2.3119m |
| honest_split_20pct_calibration_test_mean | 3.0735m | 3.2781m | +0.2046m |
| final_full_data_per_drone_quadratic_fit | 3.0251m | 3.2242m | +0.1991m |

## By Drone Type

| Drone | Attempt 5 raw | Attempt 8 raw | Attempt 5 calibrated | Attempt 8 calibrated |
|---|---:|---:|---:|---:|
| Kongsberg | 36.1536m | 33.8633m | 2.7566m | 2.7849m |
| Vestfold | 18.9774m | 16.6501m | 3.2156m | 3.5359m |

## Reading This Correctly

The 20% split score is the cleaner estimate of how calibration generalizes when only part of the real dataset is used for calibration.
The final full-data fit is useful as a best-fit mapping for this exact real dataset, but it is not a separate external-test score.
