"""Offline boundary tests for the optional adaptive Bayesian research model.
No ABE integration, no official API calls, and no production changes.
"""
import unittest
from unittest.mock import patch
import adaptive_bayesian_ensemble_max as adaptive


class AdaptiveResearchBoundaryTests(unittest.TestCase):
    def test_rank_requires_full_window(self):
        with self.assertRaisesRegex(ValueError, "Need at least 20"):
            adaptive.rank_zone([[1, 2, 3, 4, 5, 6]] * 19, 38, 10, 20)

    def test_zero_and_negative_scores_are_safely_normalized(self):
        result = adaptive.normalized({1: 0.0, 2: -4.0, 3: 0.0})
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertEqual(set(result), {1, 2, 3})
        self.assertTrue(all(v >= 0 for v in result.values()))

    def test_positive_normalization_has_unit_sum(self):
        result = adaptive.normalized({1: 2.0, 2: 3.0, 3: -4.0})
        self.assertEqual(result, {1: 0.4, 2: 0.6, 3: 0.0})

    def test_insufficient_draws_refuse_prediction(self):
        with self.assertRaisesRegex(ValueError, "At least 20"):
            adaptive.predict([{"draw_id": "115000001", "first": [1, 2, 3, 4, 5, 6], "second": 1}])

    def test_backtest_empty_has_no_invented_success_rate(self):
        result = adaptive.backtest([])
        self.assertEqual(result["tested"], 0)
        self.assertIsNone(result["first_zone_3plus_rate"])
        self.assertEqual(result["jackpot_hits"], 0)

    def test_weight_with_short_series_returns_equal_prior(self):
        self.assertEqual(adaptive.weight([1] * 20, True, 8), 0.5)

    def test_weight_respects_conservative_bounds(self):
        with patch.object(adaptive, "losses_for_previous_draws", return_value=([0.0] * 10, [1.0] * 10)):
            positive = adaptive.weight([1] * 30, True, 8)
        with patch.object(adaptive, "losses_for_previous_draws", return_value=([1.0] * 10, [0.0] * 10)):
            negative = adaptive.weight([1] * 30, True, 8)
        self.assertTrue(0 < negative < 0.5 < positive < 1)


if __name__ == "__main__":
    unittest.main()
