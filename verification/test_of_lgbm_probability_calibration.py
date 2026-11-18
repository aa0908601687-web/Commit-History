"""Causal calibration audit contracts: no test-target leakage."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_probability_calibration import (
    audit_predictions, brier, choose_alpha, shrink,
)

class ProbabilityCalibrationTests(unittest.TestCase):
    def test_fair_is_exact_brier_reference(self):
        first = list(range(1, 7))
        p = [6/38]*38
        self.assertAlmostEqual(brier(p, first), (6/38)*(1-6/38))
        self.assertEqual(shrink([0.9]*38, 6/38, 0), p)

    def test_five_holdouts_never_fit(self):
        records = [{"scores": [0.9]*38, "actual": list(range(1,7))}
                   for _ in range(5)]
        result = audit_predictions(records, 38, 6)
        self.assertEqual(result["calibration_fitted_draws"], 0)
        self.assertEqual(result["status"], "insufficient_prior_predictions")
        self.assertAlmostEqual(result["calibrated_brier"], result["fair_brier"])
        self.assertFalse(result["predictive_advantage_verified"])

    def test_only_earlier_predictions_choose_coefficients(self):
        records = [{"scores": [0.05]*38, "actual": list(range(1,7))}
                   for _ in range(32)]
        initial = audit_predictions(records, 38, 6)
        changed = [dict(x) for x in records]
        changed[-1] = {"scores": [0.99]*38, "actual": list(range(33,39))}
        updated = audit_predictions(changed, 38, 6)
        self.assertEqual(initial["per_draw"][-1]["alpha_from_prior_only"],
                         updated["per_draw"][-1]["alpha_from_prior_only"])
        self.assertEqual(initial["per_draw"][:-1], updated["per_draw"][:-1])
        self.assertEqual(initial["calibration_fitted_draws"], 2)

    def test_invalid_scores_rejected(self):
        bad = [{"scores": [float("nan")]*39, "actual": [1,2,3,4,5]}]
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit_predictions(bad, 39, 5)

    def test_alpha_disabled_without_history(self):
        self.assertEqual(choose_alpha([],38,6), 0.0)

if __name__ == "__main__":
    unittest.main()
