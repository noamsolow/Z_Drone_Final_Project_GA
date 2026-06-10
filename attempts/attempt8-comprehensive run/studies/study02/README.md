# Study02 - Nenrus Per-Drone Calibration

This study calibrates the raw Attempt 8 ensemble predictions on the real Nenrus dataset.

It intentionally mirrors Attempt 5 Study04:

- Same real Nenrus prediction task.
- Same stratified calibration/test split by `drone_type` and `true_distance_m`.
- Same calibration fractions: 10%, 20%, 30%, 50%.
- Same 100 random seeds.
- Same candidate calibration models.
- Same selected final model: `per_drone_quadratic`.

The study does not rerun depth estimation or feature extraction. It uses the raw real-data predictions produced by `studies/study01`.

Run:

```powershell
.\.venv\Scripts\python.exe "attempts\attempt8-comprehensive run\studies\study02\run_real_calibration.py"
```

Main outputs:

- `artifacts/reports/split_eval_summary.csv`
- `artifacts/reports/final_calibrated_metrics.json`
- `artifacts/reports/before_after_by_drone_type.csv`
- `artifacts/reports/comparison_vs_attempt5.csv`
- `artifacts/reports/comparison_vs_attempt5.md`
- `artifacts/plots/mae_vs_rmse_calibration_summary.png`
- `artifacts/plots/*.png`
