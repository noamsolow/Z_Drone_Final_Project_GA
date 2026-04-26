from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.calibration.features import build_regression_dataset
from pipeline.calibration.splits import make_fixed_holdout_and_cv_splits
from tests.helpers import create_sample_dataset, fake_depth_model


class SplitTests(unittest.TestCase):
    def test_holdout_and_cv_have_no_overlap_and_cover_records(self):
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=10)
            records = build_regression_dataset(
                dataset_root=dataset_root,
                depth_model=fake_depth_model,
                strict=True,
            )

            split_data = make_fixed_holdout_and_cv_splits(records, holdout_fraction=0.15, num_folds=5, seed=7)
            test_indices = set(split_data["test_indices"])
            dev_indices = set(split_data["dev_indices"])

            self.assertEqual(len(records), 120)
            self.assertEqual(len(test_indices), 18)
            self.assertEqual(len(dev_indices), 102)
            self.assertFalse(test_indices & dev_indices)
            self.assertEqual(test_indices | dev_indices, set(range(len(records))))

            all_val_indices = set()
            for fold in split_data["cv_folds"]:
                train_indices = set(fold["train_indices"])
                val_indices = set(fold["val_indices"])
                self.assertFalse(train_indices & val_indices)
                self.assertEqual(train_indices | val_indices, dev_indices)
                all_val_indices |= val_indices

            self.assertEqual(all_val_indices, dev_indices)


if __name__ == "__main__":
    unittest.main()
