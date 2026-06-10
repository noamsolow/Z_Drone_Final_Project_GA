# Study 01: Attempt8 Ensemble on Real Nenrus Data

This study applies the frozen Attempt8 `ensemble` to the prepared real-drone
Nenrus dataset without any calibration.

Protocol:

1. Export the Attempt8 aggregated RF/XGBoost ensemble from `study_all`.
2. Build the same aggregated jitter-summary feature representation for Nenrus.
3. Evaluate raw predictions overall and separately by real dataset/drone type.

No Nenrus row is used for training or calibration in this study.
