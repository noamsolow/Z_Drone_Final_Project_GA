from __future__ import annotations

import unittest

from pipeline.calibration.linear_models import (
    fit_depth_geometry_metadata_linear_model,
    fit_depth_only_linear_model,
    fit_scale_only_model,
)


class LinearModelTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "true_distance_m": 20.0,
                "relative_depth": 10.0,
                "bbox_width_px": 8.0,
                "bbox_height_px": 7.0,
                "bbox_width_norm": 0.4,
                "bbox_height_norm": 0.35,
                "bbox_area_ratio": 0.14,
                "bbox_aspect_ratio": 8.0 / 7.0,
                "bbox_center_x_norm": 0.5,
                "bbox_center_y_norm": 0.6,
                "weather": "clear_sky",
                "time_of_day": "10AM",
            },
            {
                "true_distance_m": 30.0,
                "relative_depth": 15.0,
                "bbox_width_px": 7.0,
                "bbox_height_px": 6.0,
                "bbox_width_norm": 0.35,
                "bbox_height_norm": 0.30,
                "bbox_area_ratio": 0.105,
                "bbox_aspect_ratio": 7.0 / 6.0,
                "bbox_center_x_norm": 0.45,
                "bbox_center_y_norm": 0.55,
                "weather": "light_rain",
                "time_of_day": "8PM",
            },
            {
                "true_distance_m": 40.0,
                "relative_depth": 20.0,
                "bbox_width_px": 6.0,
                "bbox_height_px": 5.0,
                "bbox_width_norm": 0.30,
                "bbox_height_norm": 0.25,
                "bbox_area_ratio": 0.075,
                "bbox_aspect_ratio": 6.0 / 5.0,
                "bbox_center_x_norm": 0.4,
                "bbox_center_y_norm": 0.5,
                "weather": "clear_sky",
                "time_of_day": "8PM",
            },
        ]

    def test_scale_only_model_matches_training_points(self):
        model = fit_scale_only_model(self.records)
        predictions = model.predict(self.records)
        self.assertAlmostEqual(predictions[0], 20.0)
        self.assertAlmostEqual(predictions[1], 30.0)
        self.assertAlmostEqual(predictions[2], 40.0)

    def test_depth_only_linear_model_has_intercept_and_predicts(self):
        model = fit_depth_only_linear_model(self.records)
        predictions = model.predict(self.records)
        self.assertEqual(predictions.shape[0], 3)
        self.assertAlmostEqual(predictions[0], 20.0, places=6)
        self.assertAlmostEqual(predictions[2], 40.0, places=6)

    def test_depth_geometry_metadata_model_exports_named_coefficients(self):
        model = fit_depth_geometry_metadata_linear_model(self.records)
        coefficient_terms = {row["term"] for row in model.coefficient_rows()}
        self.assertIn("intercept", coefficient_terms)
        self.assertIn("relative_depth", coefficient_terms)
        self.assertTrue(any(term.startswith("weather=") for term in coefficient_terms))
        self.assertTrue(any(term.startswith("time_of_day=") for term in coefficient_terms))


if __name__ == "__main__":
    unittest.main()
