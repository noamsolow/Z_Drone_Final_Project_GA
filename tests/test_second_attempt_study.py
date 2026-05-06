from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.calibration.depth_representation_features import (
    build_depth_representation_dataset,
)
from pipeline.calibration.depth_representation_study import (
    evaluate_depth_representation_methods,
)
from tests.helpers import create_sample_dataset, fake_depth_model


class SecondAttemptStudyTests(unittest.TestCase):
    def test_second_attempt_dataset_builds_expected_number_of_rows(self):
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=3)

            records, selected_samples = build_depth_representation_dataset(
                dataset_root=dataset_root,
                depth_model=fake_depth_model,
                strict=True,
                sample_count_per_stratum=2,
                sample_seed=7,
            )

            self.assertEqual(len(selected_samples), 24)
            self.assertEqual(len(records), 24 * 5 * 3)
            self.assertEqual(records[0].aggregation_method in {"inner50_median", "bbox_mean", "bbox_midpoint"}, True)

    def test_second_attempt_study_evaluates_all_method_combinations(self):
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=3)

            records, _ = build_depth_representation_dataset(
                dataset_root=dataset_root,
                depth_model=fake_depth_model,
                strict=True,
                sample_count_per_stratum=2,
                sample_seed=7,
            )
            study_result = evaluate_depth_representation_methods(records)

            self.assertEqual(len(study_result["ranking"]), 5 * 3 * 4)
            self.assertTrue(study_result["winner"]["method_id"])
            self.assertEqual(study_result["winner"]["count"], 24)
            self.assertIn("loocv_p90_absolute_error", study_result["winner"])
            self.assertIn("best_by_context", study_result)


if __name__ == "__main__":
    unittest.main()
