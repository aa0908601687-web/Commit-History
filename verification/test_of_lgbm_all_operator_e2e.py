"""Contract tests for the opt-in independent 23-operator OF-LGBM runner."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of_lgbm_all_operator_e2e as system


class AllOperatorsContractTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20261115)
        self.draws = [
            tuple(sorted(self.rng.sample(range(1, 39), 6)))
            for _ in range(50)
        ]

    def test_registry_is_exactly_23(self):
        self.assertEqual(len(system.supported_operators()), 23)

    def test_all_windows_strictly_causal(self):
        changed_far_past = list(self.draws)
        # Flip membership of 1 only in a draw older than the last 20.
        changed_far_past[0] = ((2, 3, 4, 5, 6, 7)
                               if 1 in self.draws[0] else (1, 2, 3, 4, 5, 6))
        for view in ("latest_10", "latest_20"):
            self.assertEqual(
                system.row_features(self.draws, 1, 38, view),
                system.row_features(changed_far_past, 1, 38, view),
            )
        self.assertNotEqual(
            system.row_features(self.draws, 1, 38, "full_archive")["historical_count"],
            system.row_features(changed_far_past, 1, 38, "full_archive")["historical_count"],
        )
        ten = system.row_features(self.draws, 1, 38, "latest_10")
        twenty = system.row_features(self.draws, 1, 38, "latest_20")
        archive = system.row_features(self.draws, 1, 38, "full_archive")
        self.assertNotIn("freq20", ten)
        self.assertIn("freq10", ten)
        self.assertNotIn("freq10", twenty)
        self.assertIn("freq20", twenty)
        self.assertIn("freq10", archive)
        self.assertIn("freq20", archive)
        self.assertIn("decay20", archive)
        self.assertEqual(archive["historical_draws"], float(len(self.draws)))

    def test_three_view_fusion_weights(self):
        weights = system.causal_view_weights(self.draws, 38)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(set(weights), set(system.VIEW_ANCHORS))
        self.assertTrue(all(0 < w < 1 for w in weights.values()))
        # Future target must not influence weights for a current forecast.
        self.assertEqual(weights, system.causal_view_weights(list(self.draws), 38))
        changed = list(self.draws)
        changed[-1] = tuple(range(1, 7))
        self.assertEqual(system.causal_view_weights(self.draws[:-1], 38),
                         system.causal_view_weights(changed[:-1], 38))

    def test_chronological_anchor_boundaries(self):
        self.assertEqual(system.anchors(50, "latest_10"), list(range(40, 50)))
        self.assertEqual(system.anchors(50, "latest_20"), list(range(30, 50)))
        self.assertEqual(system.anchors(50, "full_archive"), list(range(20, 50)))
        # Verify full-history training never silently shrinks to a 100-anchor sample.
        self.assertEqual(system.anchors(250, "full_archive"), list(range(20, 250)))
        self.assertEqual(system.anchors(20, "full_archive"), [])

    def test_categorical_columns_are_real_categories(self):
        frame = system.feature_frame([
            system.row_features(self.draws, n, 38, "full_archive")
            for n in range(1, 39)
        ])
        for col in system.CATEGORY_COLUMNS:
            self.assertEqual(frame[col].dtype.name, "category")
        self.assertGreaterEqual(len(frame.select_dtypes(include="number").columns), 2)


if __name__ == "__main__":
    unittest.main()
