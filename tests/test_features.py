from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.calibration.features import build_regression_dataset
from tests.helpers import create_sample_dataset, fake_depth_model


class FeatureBuilderTests(unittest.TestCase):
    def test_build_regression_dataset_computes_expected_geometry(self):
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=1)

            records = build_regression_dataset(
                dataset_root=dataset_root,
                depth_model=fake_depth_model,
                strict=True,
                max_samples=1,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.weather, "clear_sky")
            self.assertEqual(record.time_of_day, "10AM")
            self.assertAlmostEqual(record.true_distance_m, 30.0)
            self.assertEqual(record.bbox_width_px, 8)
            self.assertEqual(record.bbox_height_px, 8)
            self.assertAlmostEqual(record.bbox_width_norm, 8.0 / 24.0)
            self.assertAlmostEqual(record.bbox_height_norm, 8.0 / 24.0)
            self.assertAlmostEqual(record.bbox_area_ratio, 64.0 / (24.0 * 24.0))
            self.assertAlmostEqual(record.bbox_aspect_ratio, 1.0)
            self.assertAlmostEqual(record.bbox_center_x_norm, 10.0 / 24.0)
            self.assertAlmostEqual(record.bbox_center_y_norm, 10.0 / 24.0)
            self.assertAlmostEqual(record.relative_depth, 30.0)


if __name__ == "__main__":
    unittest.main()
