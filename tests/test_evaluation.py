from __future__ import annotations

import unittest

from pipeline.calibration.evaluation import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_evaluate_predictions_returns_overall_and_per_range_metrics(self):
        records = [
            {"true_distance_m": 30.0, "weather": "clear_sky", "time_of_day": "10AM"},
            {"true_distance_m": 80.0, "weather": "clear_sky", "time_of_day": "10AM"},
            {"true_distance_m": 125.0, "weather": "light_rain", "time_of_day": "8PM"},
        ]
        predictions = [27.0, 88.0, 120.0]
        evaluation = evaluate_predictions(records, predictions)

        self.assertIn("overall", evaluation)
        self.assertIn("per_distance_range", evaluation)
        self.assertAlmostEqual(evaluation["overall"]["mae"], (3.0 + 8.0 + 5.0) / 3.0)
        self.assertIn("median_absolute_error", evaluation["overall"])
        self.assertIn("within_10m_rate", evaluation["overall"])
        self.assertEqual(evaluation["per_distance_range"]["near"]["count"], 1)
        self.assertEqual(evaluation["per_distance_range"]["mid"]["count"], 1)
        self.assertEqual(evaluation["per_distance_range"]["far"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
