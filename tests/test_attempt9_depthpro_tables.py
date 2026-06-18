from __future__ import annotations

import csv
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from pipeline.calibration.linear_models import GEOMETRY_FEATURE_NAMES
from tests.helpers import create_sample_dataset


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load module from {}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constant_depth_model(image):
    if isinstance(image, Image.Image):
        width, height = image.size
        return np.full((height, width), 11.0, dtype=np.float32)
    array = np.asarray(image)
    return np.full((array.shape[0], array.shape[1]), 11.0, dtype=np.float32)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return dict(next(csv.DictReader(handle)))


class Attempt9DepthProTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study02 = _load_module(
            "attempt9_study02_run_extract_fused_features",
            "attempts/attempt9/study02/run_extract_fused_features.py",
        )
        cls.study03 = _load_module(
            "attempt9_study03_run_extract_noisy_aggregated_features",
            "attempts/attempt9/study03/run_extract_noisy_aggregated_features.py",
        )

    def test_study02_keeps_fused_schema_and_recomputes_depth_columns(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=1)
            image_path = next(dataset_root.glob("*.png"))
            label_path = image_path.with_suffix(".txt")

            source_csv = root / "source_fused.csv"
            fieldnames = [
                "image_path",
                "label_path",
                "stratum_key",
                "selection_rank_within_stratum",
                "true_distance_m",
                "weather",
                "time_of_day",
                "bbox_only__inner50_median__object_depth",
            ]
            _write_csv(
                source_csv,
                fieldnames,
                [
                    {
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "stratum_key": "30.0|clear_sky|10AM",
                        "selection_rank_within_stratum": 1,
                        "true_distance_m": 30.0,
                        "weather": "clear_sky",
                        "time_of_day": "10AM",
                        "bbox_only__inner50_median__object_depth": 1.0,
                    }
                ],
            )

            result = self.study02.run_feature_extraction(
                config={
                    "sources": {"source_fused_features_csv": str(source_csv)},
                    "max_samples": None,
                    "study": {
                        "context_windows": [{"name": "bbox_only", "scale": 1.0}],
                        "aggregation_methods": ["inner50_median"],
                        "score_fields": ["object_depth"],
                        "resume_from_partial": False,
                    },
                    "output_root": str(root / "out"),
                },
                depth_model=constant_depth_model,
                verbose=False,
            )

            row = _read_first_row(Path(result["fused_features_csv"]))
            self.assertEqual(list(row.keys()), fieldnames)
            self.assertEqual(row["weather"], "clear_sky")
            self.assertAlmostEqual(float(row["bbox_only__inner50_median__object_depth"]), 11.0)

    def test_study03_keeps_aggregated_schema_and_recomputes_jitter_depth_columns(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            create_sample_dataset(dataset_root, repeats_per_stratum=1)
            image_path = next(dataset_root.glob("*.png"))
            label_path = image_path.with_suffix(".txt")

            geometry_columns = []
            for name in GEOMETRY_FEATURE_NAMES:
                geometry_columns.append("{}__jitter_median".format(name))
                geometry_columns.append("{}__jitter_std".format(name))
            depth_columns = [
                "bbox_only__inner50_median__object_depth__jitter_median",
                "bbox_only__inner50_median__object_depth__jitter_std",
            ]
            fieldnames = [
                "image_path",
                "label_path",
                "stratum_key",
                "true_distance_m",
                "weather",
                "time_of_day",
                "num_jitter_boxes",
            ] + geometry_columns + depth_columns
            source_row = {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "stratum_key": "30.0|clear_sky|10AM",
                "true_distance_m": 30.0,
                "weather": "clear_sky",
                "time_of_day": "10AM",
                "num_jitter_boxes": 1,
                "bbox_only__inner50_median__object_depth__jitter_median": 1.0,
                "bbox_only__inner50_median__object_depth__jitter_std": 2.0,
            }
            for column in geometry_columns:
                source_row[column] = 123.0

            source_csv = root / "source_aggregated.csv"
            _write_csv(source_csv, fieldnames, [source_row])

            result = self.study03.run_feature_extraction(
                config={
                    "sources": {"source_aggregated_features_csv": str(source_csv)},
                    "max_samples": None,
                    "study": {
                        "context_windows": [{"name": "bbox_only", "scale": 1.0}],
                        "aggregation_methods": ["inner50_median"],
                        "jitter_boxes": [{"name": "original", "dx_frac": 0.0, "dy_frac": 0.0, "scale": 1.0}],
                        "resume_from_partial": False,
                    },
                    "output_root": str(root / "out"),
                },
                depth_model=constant_depth_model,
                verbose=False,
            )

            row = _read_first_row(Path(result["aggregated_features_csv"]))
            self.assertEqual(list(row.keys()), fieldnames)
            self.assertAlmostEqual(float(row["bbox_width_px__jitter_median"]), 123.0)
            self.assertAlmostEqual(float(row["bbox_only__inner50_median__object_depth__jitter_median"]), 11.0)
            self.assertAlmostEqual(float(row["bbox_only__inner50_median__object_depth__jitter_std"]), 0.0)


if __name__ == "__main__":
    unittest.main()
