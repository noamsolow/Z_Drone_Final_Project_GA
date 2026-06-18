# Poster Model Table

| Rank | Model | Short explanation | Test MAE |
| ---: | --- | --- | ---: |
| 1 | ensemble | A tuned blend of RF+jitter and XGBoost+jitter; the final best poster model. | 6.341m |
| 2 | XGBoost+jitter | A boosted-tree model trained on the same aggregated jitter-summary feature table. | 6.355m |
| 3 | RF+jitter | A random forest trained on aggregated jitter-summary features for more robust bbox/depth signals. | 6.714m |
| 4 | RF | A random forest trained on exact depth, bbox geometry, and metadata features. | 7.543m |
| 5 | improved linear | A stacked linear model that adds a lower-stage RF prediction as an extra feature. | 11.943m |
| 6 | linear+bb | A linear model that adds bbox geometry and image metadata to the depth signal. | 13.353m |
| 7 | depth linear | A linear model using relative depth as the main predictor. | 32.299m |
| 8 | scale only | A minimal calibration baseline that learns a global scale from the raw depth signal. | 77.383m |
