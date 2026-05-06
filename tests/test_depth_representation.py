from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from pipeline.depth.representation import (
    ContextWindowSpec,
    compute_depth_representation,
    extract_context_crop,
)


class DepthRepresentationTests(unittest.TestCase):
    def test_extract_context_crop_translates_bbox_into_crop_coordinates(self):
        image = Image.new("RGB", (10, 10), color=(0, 0, 0))
        bbox = (2, 2, 6, 6)

        crop_data = extract_context_crop(
            image=image,
            bbox=bbox,
            context_spec=ContextWindowSpec(name="bbox_expand_2x", scale=2.0),
        )

        self.assertEqual(crop_data["crop_bbox"], (0, 0, 8, 8))
        self.assertEqual(crop_data["local_bbox"], (2, 2, 6, 6))
        self.assertEqual(crop_data["image"].size, (8, 8))

    def test_compute_depth_representation_supports_all_requested_aggregations(self):
        depth_map = np.zeros((10, 10), dtype=np.float32)
        depth_map[2:6, 2:6] = 10.0
        depth_map[3:5, 3:5] = 20.0
        bbox = (2, 2, 6, 6)

        inner50 = compute_depth_representation(depth_map, bbox, aggregation_method="inner50_median")
        bbox_mean = compute_depth_representation(depth_map, bbox, aggregation_method="bbox_mean")
        midpoint = compute_depth_representation(depth_map, bbox, aggregation_method="bbox_midpoint")

        self.assertAlmostEqual(inner50["object_depth"], 20.0)
        self.assertAlmostEqual(bbox_mean["object_depth"], 12.5)
        self.assertAlmostEqual(midpoint["object_depth"], 20.0)

        self.assertAlmostEqual(inner50["surrounding_ring_depth"], 0.0)
        self.assertAlmostEqual(bbox_mean["surrounding_ring_depth"], 0.0)
        self.assertAlmostEqual(midpoint["surrounding_ring_depth"], 0.0)

        self.assertGreater(inner50["object_minus_ring"], 0.0)
        self.assertGreater(bbox_mean["object_minus_ring"], 0.0)
        self.assertGreater(midpoint["object_minus_ring"], 0.0)
        self.assertGreaterEqual(inner50["object_depth_percentile_5_95"], 0.0)
        self.assertLessEqual(inner50["object_depth_percentile_5_95"], 1.0)


if __name__ == "__main__":
    unittest.main()
