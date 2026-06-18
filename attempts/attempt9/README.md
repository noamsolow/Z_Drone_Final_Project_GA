# Attempt 9: Depth Pro Dataset Rebuild

Attempt 9 rebuilds the feature sources used by Attempt 8, while replacing the
Depth Anything values with Apple Depth Pro metric depth values.

For now, these ambiguous folders are intentionally excluded from all generated
feature tables:

- `depth_20/clear_sky/10AM`
- `depth_20/light_rain/10AM`
- `depth_30/clear_sky/10AM`
- `depth_30/light_rain/10AM`

Expected clean row counts before recovering those folders:

- `study01`: `13192`
- `study02`: `13391`
- `study03`: `13391`

Run order:

```powershell
powershell -ExecutionPolicy Bypass -File attempts\attempt9\study01\run_extract_features.ps1 2>&1 | Tee-Object -FilePath attempts\attempt9\study01\artifacts\run_extract_features.log
powershell -ExecutionPolicy Bypass -File attempts\attempt9\study02\run_extract_fused_features.ps1 2>&1 | Tee-Object -FilePath attempts\attempt9\study02\artifacts\run_extract_fused_features.log
powershell -ExecutionPolicy Bypass -File attempts\attempt9\study03\run_extract_noisy_aggregated_features.ps1 2>&1 | Tee-Object -FilePath attempts\attempt9\study03\artifacts\run_extract_noisy_aggregated_features.log
```

Or run all dataset builders:

```powershell
powershell -ExecutionPolicy Bypass -File attempts\attempt9\run_build_depthpro_datasets.ps1
```

After the three Depth Pro feature tables exist, rerun the Attempt 8 comparison:

```powershell
powershell -ExecutionPolicy Bypass -File attempts\attempt9\study_all\run_attempt8_depthpro.ps1 2>&1 | Tee-Object -FilePath attempts\attempt9\study_all\artifacts\run_attempt8_depthpro.log
```

Outputs:

- `study01/artifacts/features/features_available.csv`: Attempt1-compatible exact dataset.
- `study02/artifacts/fused_features/depth_only_feature_table.csv`: Study06-compatible fused depth table.
- `study03/artifacts/features/noisy_aggregated_feature_table.csv`: Study03-compatible jitter aggregation table.
- `study_all/artifacts`: Attempt8 model comparison results using the Depth Pro feature sources.
