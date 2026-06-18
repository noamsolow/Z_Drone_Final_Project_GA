from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from pipeline.calibration.features import (
    build_regression_dataset,
    read_regression_records_csv,
    write_regression_records_csv,
)
from tests.helpers import create_sample_dataset, fake_depth_model


def _load_attempt9_study01_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "attempts" / "attempt9" / "study01" / "run_extract_features.py"
    spec = importlib.util.spec_from_file_location("attempt9_study01_run_extract_features", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Attempt 9 Study 01 module from {}".format(module_path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constant_depth_model(image):
    if isinstance(image, Image.Image):
        width, height = image.size
        return np.full((height, width), 7.0, dtype=np.float32)

    array = np.asarray(image)
    return np.full((array.shape[0], array.shape[1]), 7.0, dtype=np.float32)


class Attempt9Study01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_attempt9_study01_module()

    def test_rebuild_keeps_attempt1_schema_and_updates_only_depth_values(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=1)

            base_records = build_regression_dataset(
                dataset_root=dataset_root,
                depth_model=fake_depth_model,
                strict=True,
            )
            base_csv = root / "base_features.csv"
            write_regression_records_csv(base_records, base_csv)

            output_root = root / "attempt9_artifacts"
            config = {
                "study_name": "attempt9_test",
                "dataset_root": str(dataset_root),
                "strict": True,
                "max_samples": 3,
                "sources": {
                    "base_features_csv": str(base_csv),
                },
                "depth_model": {
                    "kind": "depth_pro",
                    "checkpoint_path": "unused-in-tests.pt",
                    "device": "cpu",
                },
                "study": {
                    "progress_log_every_rows": 2,
                },
                "output_root": str(output_root),
            }

            result = self.module.run_feature_extraction(
                config=config,
                depth_model=constant_depth_model,
                verbose=False,
            )

            rebuilt_records = read_regression_records_csv(Path(result["features_csv"]))
            self.assertEqual(len(rebuilt_records), 3)

            for base_record, rebuilt_record in zip(base_records[:3], rebuilt_records):
                self.assertEqual(base_record.image_path, rebuilt_record.image_path)
                self.assertEqual(base_record.label_path, rebuilt_record.label_path)
                self.assertEqual(base_record.true_distance_m, rebuilt_record.true_distance_m)
                self.assertEqual(base_record.bbox_width_px, rebuilt_record.bbox_width_px)
                self.assertEqual(base_record.bbox_height_px, rebuilt_record.bbox_height_px)
                self.assertEqual(base_record.bbox_width_norm, rebuilt_record.bbox_width_norm)
                self.assertEqual(base_record.bbox_height_norm, rebuilt_record.bbox_height_norm)
                self.assertEqual(base_record.bbox_area_ratio, rebuilt_record.bbox_area_ratio)
                self.assertEqual(base_record.bbox_aspect_ratio, rebuilt_record.bbox_aspect_ratio)
                self.assertEqual(base_record.bbox_center_x_norm, rebuilt_record.bbox_center_x_norm)
                self.assertEqual(base_record.bbox_center_y_norm, rebuilt_record.bbox_center_y_norm)
                self.assertEqual(base_record.weather, rebuilt_record.weather)
                self.assertEqual(base_record.time_of_day, rebuilt_record.time_of_day)
                self.assertNotEqual(base_record.relative_depth, rebuilt_record.relative_depth)
                self.assertAlmostEqual(rebuilt_record.relative_depth, 7.0)


if __name__ == "__main__":
    unittest.main()
