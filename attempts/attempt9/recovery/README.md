# Attempt 9 Recovery: Problematic Folders

This folder builds separate supplement datasets for the four ambiguous folders
that were excluded from the clean Attempt 9 datasets:

- `depth_20/clear_sky/10AM`
- `depth_20/light_rain/10AM`
- `depth_30/clear_sky/10AM`
- `depth_30/light_rain/10AM`

The recovery script recomputes rows from the images and labels currently on
disk. It does not merge them into the clean datasets automatically.

Run all three supplements:

```powershell
powershell -ExecutionPolicy Bypass -File attempts\attempt9\recovery\run_build_problematic_folder_supplements.ps1 2>&1 | Tee-Object -FilePath attempts\attempt9\recovery\artifacts\run_build_problematic_folder_supplements.log
```

Run one supplement at a time:

```powershell
powershell -ExecutionPolicy Bypass -File attempts\attempt9\recovery\run_build_problematic_folder_supplements.ps1 -Datasets study01
powershell -ExecutionPolicy Bypass -File attempts\attempt9\recovery\run_build_problematic_folder_supplements.ps1 -Datasets study02
powershell -ExecutionPolicy Bypass -File attempts\attempt9\recovery\run_build_problematic_folder_supplements.ps1 -Datasets study03
```

Expected target samples from the current disk folders: `1673`.

Outputs:

- `artifacts/study01/features_problematic_folders.csv`
- `artifacts/study02/depth_only_feature_table_problematic_folders.csv`
- `artifacts/study03/noisy_aggregated_feature_table_problematic_folders.csv`
- `artifacts/audit/problematic_folder_manifest.csv`
- `artifacts/summary.json`
