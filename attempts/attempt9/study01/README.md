# Attempt 9 / Study 01

This study rebuilds the current Attempt1-style feature dataset with the same
row schema and row ordering, but replaces the old Depth Anything depth value
with a new Depth Pro depth measurement.

Key design choices:

- Keep the output CSV compatible with the existing downstream code.
- Reuse all non-depth columns from the base Attempt1 feature table.
- Recompute only the depth value from the original image + YOLO bbox.
- Print progress to the console during extraction so long runs are visible.

Main files:

- `config.yaml`
- `run_extract_features.py`

Expected output:

- `attempts/attempt9/study01/artifacts/features/features.csv`
- `attempts/attempt9/study01/artifacts/features/summary.json`
