# Attempt 10: Shallow MLP Feature Baseline

This calculation trains a small one-hidden-layer MLP on the final Attempt 8
21-feature representation and evaluates it on the same held-out test split.

## Inputs
- Feature table: `C:\Users\noams\OneDrive\Desktop\school\final project\Z_Drone_Final_Project_GA\attempts\attempt8-comprehensive run\studies\study_all\artifacts\features\aggregated_common_feature_table.csv`
- Split: `C:\Users\noams\OneDrive\Desktop\school\final project\Z_Drone_Final_Project_GA\attempts\attempt8-comprehensive run\studies\study_all\artifacts\splits\unified_split.json`
- Feature schema: `C:\Users\noams\OneDrive\Desktop\school\final project\Z_Drone_Final_Project_GA\attempts\attempt8-comprehensive run\studies\study01\artifacts\reports\model_schema.json`
- Feature count: `21`

## Result
- Selected MLP seed: `20260520`
- MLP test MAE: `7.596` m
- MLP test RMSE: `10.383` m
- MLP test R2: `0.926`
- MLP within 10 m: `72.0%`
- Attempt 8 final ensemble test MAE: `6.008` m
- Attempt 8 final ensemble within 10 m: `81.0%`

The MLP improves over the linear baselines but remains worse than the final
RF--XGBoost ensemble on the same feature space.

## Descriptor probe

A ridge leave-one-feature-out probe was added as a lightweight descriptor-level
sanity check. The largest MAE increases come from removing:
- `bbox_only__inner50_median__object_depth__jitter_median`: delta MAE `0.683` m
- `bbox_area_ratio__jitter_median`: delta MAE `0.337` m
- `bbox_aspect_ratio__jitter_median`: delta MAE `0.321` m
- `bbox_center_y_norm__jitter_std`: delta MAE `0.050` m
- `bbox_aspect_ratio__jitter_std`: delta MAE `0.044` m
