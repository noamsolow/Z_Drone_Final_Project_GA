# Attempt6 Study02: BBox Scale Domain Randomization

This study tests whether broad synthetic bbox scale randomization improves
generalization to real drone images.

## Motivation

Attempt6 Study01 showed that reducing geometry improved real external
performance, but simple geometry dropout/noise did not. The remaining issue was
still strong overprediction on real images.

The hypothesis here is that the model does not just need noisy geometry. It
needs to see many possible object physical scales during synthetic training, so
it does not treat bbox size as a single fixed proxy for distance.

## Important Constraint

The model is trained only on synthetic rows.

The real Nenrus rows are used only for external evaluation. The scale ranges in
this study are generic domain-randomization settings, not fitted from Nenrus
statistics.

## Variants

- `baseline_full_geometry`
- `reduced_geometry`
- `scale_uniform_060_140_full_geometry`
- `scale_uniform_040_140_full_geometry`
- `scale_uniform_040_140_reduced_geometry`
- `scale_lognormal_full_geometry`
- `scale_lognormal_reduced_geometry`
- `raw_depth_only`

## Outputs

Reports:

- `artifacts/reports/variant_metrics.csv`
- `artifacts/reports/predictions.csv`
- `artifacts/reports/real_by_drone_type.csv`
- `artifacts/reports/real_by_distance.csv`
- `artifacts/reports/feature_importances.csv`
- `artifacts/reports/feature_bucket_importances.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/analysis.md`

Plots:

- `artifacts/plots/real_external_overall_by_variant.png`
- `artifacts/plots/synthetic_vs_real_mae_by_variant.png`
- `artifacts/plots/real_mae_by_drone_type_and_variant.png`
- `artifacts/plots/feature_bucket_importances_by_variant.png`
- `artifacts/plots/real_overprediction_by_variant.png`
- `artifacts/plots/winner_real_true_vs_predicted.png`

## Run

Dry run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study02/run_study_02.py" --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe "attempts/attempt6/studies/study02/run_study_02.py"
```
