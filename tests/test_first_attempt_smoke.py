from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from first_attempt.run_extract_features import run_feature_extraction
from first_attempt.run_study import run_study
from tests.helpers import create_sample_dataset, create_temp_config, fake_depth_model


class FirstAttemptSmokeTests(unittest.TestCase):
    def test_full_first_attempt_workflow_writes_expected_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            output_root = root / "artifacts"
            create_sample_dataset(dataset_root, repeats_per_stratum=10)
            config = create_temp_config(dataset_root=dataset_root, output_root=output_root)

            extract_result = run_feature_extraction(config=config, depth_model=fake_depth_model)
            self.assertEqual(extract_result["num_records"], 120)

            study_result = run_study(config=config)
            self.assertIn("winner_model_name", study_result)

            expected_paths = [
                output_root / "features" / "features.csv",
                output_root / "features" / "summary.json",
                output_root / "splits" / "test_split.json",
                output_root / "splits" / "cv_folds.json",
                output_root / "reports" / "cv_metrics.json",
                output_root / "reports" / "test_metrics.json",
                output_root / "reports" / "correlations.csv",
                output_root / "reports" / "grouped_error_summary.csv",
                output_root / "reports" / "model_coefficients.csv",
                output_root / "reports" / "test_predictions.csv",
            ]
            for path in expected_paths:
                self.assertTrue(path.exists(), str(path))

            cv_metrics = json.loads((output_root / "reports" / "cv_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("winner_model_name", cv_metrics)


if __name__ == "__main__":
    unittest.main()
