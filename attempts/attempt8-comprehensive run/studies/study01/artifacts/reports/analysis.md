# Study 01: Raw Attempt8 Ensemble on Real Nenrus Data

## Protocol

- The Attempt8 ensemble was exported from `study_all` and trained on all synthetic StudyAll rows.
- The real Nenrus rows were used only for external evaluation.
- No calibration, fine-tuning, or real-domain retraining was applied.

## Overall Raw Result

- Count: `489`
- MAE: `23.796m`
- RMSE: `28.248m`
- R2: `-1.101`
- Mean signed error: `23.795m`
- Within 10m: `0.160`
- Overprediction rate: `0.998`

## By Real Dataset / Drone Type

- `Kongsberg`: count `203`, MAE `33.863m`, mean signed error `33.863m`, within 10m `0.000`
- `Vestfold`: count `286`, MAE `16.650m`, mean signed error `16.649m`, within 10m `0.273`

## By Distance

- `4.0`m: count `18`, MAE `15.276m`, mean signed error `15.276m`
- `5.0`m: count `6`, MAE `17.284m`, mean signed error `17.284m`
- `9.0`m: count `26`, MAE `13.580m`, mean signed error `13.580m`
- `10.0`m: count `19`, MAE `9.996m`, mean signed error `9.996m`
- `15.0`m: count `44`, MAE `14.215m`, mean signed error `14.215m`
- `20.0`m: count `46`, MAE `19.095m`, mean signed error `19.095m`
- `25.0`m: count `42`, MAE `21.645m`, mean signed error `21.645m`
- `30.0`m: count `35`, MAE `27.088m`, mean signed error `27.088m`
- `35.0`m: count `47`, MAE `26.559m`, mean signed error `26.559m`
- `40.0`m: count `37`, MAE `28.869m`, mean signed error `28.869m`
- `45.0`m: count `39`, MAE `38.622m`, mean signed error `38.615m`
- `50.0`m: count `42`, MAE `32.631m`, mean signed error `32.631m`
- `55.0`m: count `20`, MAE `16.701m`, mean signed error `16.701m`
- `60.0`m: count `13`, MAE `10.651m`, mean signed error `10.651m`
- `65.0`m: count `19`, MAE `18.613m`, mean signed error `18.613m`
- `70.0`m: count `12`, MAE `36.758m`, mean signed error `36.758m`
- `75.0`m: count `24`, MAE `37.212m`, mean signed error `37.212m`

## Exported Model

- Ensemble: `C:\Users\depthlev\Desktop\Z_Drone_Final_Project_GA\attempts\attempt8-comprehensive run\studies\study01\artifacts\models\full_study_all_dataset\ensemble.joblib`
- Training rows: `15064`
- RF candidate: `rf_deep`
- XGB candidate: `xgb_deep`
- Blend weights: `{'rf_weight': 0.17, 'xgb_weight': 0.83}`
