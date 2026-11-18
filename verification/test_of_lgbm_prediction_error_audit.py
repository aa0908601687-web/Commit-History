"""Regression tests for fair-null prediction error evaluation (not forecasting)."""
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_prediction_error_audit import (
    binomial_tail, hit_probabilities, summarize_outcomes, wilson_interval,
)


class PredictionErrorAuditTests(unittest.TestCase):
    def test_exact_distribution_sums_to_one_and_mean(self):
        for maximum, count in [(38, 6), (49, 6), (39, 5)]:
            probs = hit_probabilities(maximum, count)
            self.assertAlmostEqual(sum(probs.values()), 1)
            self.assertAlmostEqual(sum(k*probs[str(k)] for k in range(count+1)),
                                   count*count/maximum)

    def test_exact_binomial_tail(self):
        self.assertAlmostEqual(binomial_tail(5, 0, 0.2), 1)
        self.assertAlmostEqual(binomial_tail(5, 5, 0.2), 0.2**5)
        self.assertAlmostEqual(binomial_tail(5, 1, 0.2), 1 - 0.8**5)

    def test_small_sample_not_claimed_predictive(self):
        rows = [{"matches": n, "second_pick": 2, "second_actual": a}
                for n, a in [(0, 1), (1, 2), (1, 5), (2, 7), (0, 8)]]
        report = summarize_outcomes(rows, 38, 6, 8)
        self.assertEqual(report["observed_3plus_count"], 0)
        self.assertEqual(report["power_second_zone"]["matched_draws"], 1)
        self.assertAlmostEqual(report["power_second_zone"]["fair_baseline_rate"], 1/8)
        self.assertFalse(report["statistically_verified_predictive_advantage"])
        self.assertLessEqual(report["observed_3plus_rate_wilson_95pct"][1], 1.0)
        self.assertGreater(report["observed_3plus_rate_wilson_95pct"][1], 0)

    def test_539_has_no_second_zone(self):
        rows = [{"matches": 0}, {"matches": 3}]
        report = summarize_outcomes(rows, 39, 5)
        self.assertNotIn("power_second_zone", report)
        self.assertEqual(report["observed_3plus_count"], 1)

    def test_rejects_unusable_second_zone(self):
        with self.assertRaisesRegex(ValueError, "second zone"):
            summarize_outcomes([{"matches": 1, "second_pick": None,
                                 "second_actual": 2}], 38, 6, 8)

    def test_invalid_probability_and_interval_rejected(self):
        with self.assertRaises(ValueError):
            binomial_tail(3, 4, 0.1)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


if __name__ == "__main__":
    unittest.main()
