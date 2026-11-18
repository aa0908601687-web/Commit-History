"""Regression contract for the requested OF-LGBM learned three-view update.

These tests are deliberately red against the pre-migration runner. They should
be made green by the implementation, not weakened or skipped.
"""
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of_lgbm_all_operator_e2e as system


class LearnedFusionContract(unittest.TestCase):
    def test_selection_does_not_use_brier_frequency_weights(self):
        source = inspect.getsource(system.run)
        self.assertNotIn(
            "causal_view_weights(", source,
            "Brier-derived historical-frequency weights must not select tickets",
        )
        self.assertNotIn(
            "causal_frequency_brier_calibration_not_ML", source,
            "The output must not advertise Brier-driven ticket fusion",
        )

    def test_three_historical_perspectives_remain(self):
        self.assertEqual(
            system.VIEW_ANCHORS,
            {"latest_10": 10, "latest_20": 20, "full_archive": None},
        )

    def test_ticket_pair_score_is_not_silently_removed(self):
        from of_lgbm_ticket_selection import PAIR_WEIGHT
        self.assertEqual(PAIR_WEIGHT, 0.05)


if __name__ == "__main__":
    unittest.main()
