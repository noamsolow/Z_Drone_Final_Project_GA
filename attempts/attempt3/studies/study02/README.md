# Study 02: Stacked Big Random-Forest Comparison

This study is the second step of `attempt3`.

It keeps the same core idea as Study 01:

- reuse the old attempt 1 geometry and metadata setup
- reuse the Study 06 small depth-only random forest as a learned depth signal

But it changes the stage-two model family:

- Study 01 used linear regression
- Study 02 uses a larger random forest

The purpose is to test whether a non-linear top model can extract more value
from:

- `depth_lower_model_prediction`
- bbox geometry
- weather
- time of day

## Main Risks Addressed

This implementation explicitly handles the main methodological risks.

### Leakage

The lower RF is cross-fit inside each outer split.

That means the upper model never receives a lower-model prediction produced by
a lower RF that was trained on that same row.

### Overfitting

The upper model does not assume that an unbounded forest is automatically best.

Instead, the study evaluates several candidate RF settings that vary:

- `max_depth`
- `min_samples_leaf`
- `max_features`

and chooses the best candidate by CV performance on the development set.

### Runtime waste

The lower-model predictions are cached per outer fold and reused across all
upper-model candidates, so the nested setup stays honest without recomputing
the same lower-model work again and again.

## Main Outputs

When the study is run, it writes:

- `artifacts/features/merged_feature_table.csv`
- `artifacts/features/summary.json`
- `artifacts/splits/test_split.json`
- `artifacts/splits/cv_folds.json`
- `artifacts/reports/cv_metrics.json`
- `artifacts/reports/test_metrics.json`
- `artifacts/reports/cv_predictions.csv`
- `artifacts/reports/test_predictions.csv`
- `artifacts/reports/grouped_error_summary.csv`
- `artifacts/reports/feature_correlations.csv`
- `artifacts/reports/upper_rf_candidate_metrics.csv`
- `artifacts/reports/upper_rf_selected_models.csv`
- `artifacts/reports/upper_rf_feature_importances.csv`
- `artifacts/reports/lower_model_feature_importances.csv`
- `artifacts/reports/lower_model_summary.json`
- `artifacts/reports/summary.json`

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt3/studies/study02/run_study_02.py"
```
