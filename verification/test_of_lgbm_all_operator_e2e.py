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
            for _ in range(20)
        ]

    def test_registry_is_exactly_23(self):
        self.assertEqual(len(system.supported_operators()), 23)

    def test_all_windows_strictly_causal(self):
        for view in system.VIEW_ANCHORS:
            features = system.row_features(self.draws, 1, 38, view)
            changed_past = list(self.draws)
            changed_past[0] = (1, 2, 3, 4, 5, 6)
            self.assertEqual(features, system.row_features(changed_past, 1, 38, view))
        five = system.row_features(self.draws, 1, 38, "latest_5")
        ten = system.row_features(self.draws, 1, 38, "latest_10")
        archive = system.row_features(self.draws, 1, 38, "full_archive")
        self.assertNotIn("freq10", five)
        self.assertNotIn("freq5", ten)
        self.assertIn("freq10", archive)
        self.assertIn("freq5", archive)
        self.assertIn("decay10", archive)

    def test_three_view_fusion_weights(self):
        self.assertAlmostEqual(sum(system.VIEW_WEIGHTS.values()), 1.0)
        self.assertEqual(set(system.VIEW_WEIGHTS), set(system.VIEW_ANCHORS))

    def test_chronological_anchor_boundaries(self):
        self.assertEqual(system.anchors(20, "latest_5"), [15, 16, 17, 18, 19])
        self.assertEqual(system.anchors(20, "latest_10"), list(range(10, 20)))
        self.assertEqual(system.anchors(20, "full_archive"), list(range(10, 20)))

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
