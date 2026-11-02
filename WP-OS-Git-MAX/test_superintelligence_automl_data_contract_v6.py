"""Offline coverage for isolated AutoML data preparation; no TensorFlow needed."""
import csv
import tempfile
import unittest
from pathlib import Path

import superintelligence_automl_research as automl


class AutoMLDataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "draws.csv"
        self.draws = [
            {"draw_id": f"115000{i:03d}", "first": [1, 2, 3, 4, 5, 6],
             "second": 1 + i % 8}
            for i in range(1, 13)
        ]

    def write(self, draws):
        with self.path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(("draw_id", "first", "second"))
            for draw in draws:
                writer.writerow((draw["draw_id"], " ".join(map(str, draw["first"])),
                                 draw["second"]))

    def test_load_and_dataset_are_leakage_aware(self):
        self.write(self.draws)
        loaded = automl.load_draws(self.path)
        self.assertEqual(loaded, self.draws)
        x, first_targets, second_targets = automl.dataset(loaded)
        self.assertEqual((len(x), len(first_targets), len(second_targets)), (2, 2, 2))
        self.assertEqual(len(x[0]), 92)
        self.assertEqual(x[0], automl.features(loaded[:10]))
        self.assertEqual(first_targets[0][0:6], [1.0] * 6)
        self.assertEqual(sum(first_targets[0]), 6)
        self.assertEqual(second_targets[0], loaded[10]["second"] - 1)

    def test_frequency_features_use_only_latest_5_and_10(self):
        self.assertEqual(len(automl.features(self.draws)), 92)
        changed = [dict(row) for row in self.draws]
        changed[0] = {"draw_id": "115000001", "first": [7, 8, 9, 10, 11, 12],
                      "second": 8}
        self.assertEqual(automl.features(changed), automl.features(self.draws))
        f = automl.features(self.draws)
        self.assertAlmostEqual(f[0], 1.0)
        self.assertAlmostEqual(f[38], sum(x["second"] == 1 for x in self.draws[-5:]) / 5)

    def test_short_history_rejected(self):
        self.write(self.draws[:11])
        with self.assertRaisesRegex(ValueError, "At least 12"):
            automl.load_draws(self.path)

    def test_duplicate_and_out_of_range_first_zone_rejected(self):
        for first in ([1, 1, 3, 4, 5, 6], [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5, 39]):
            bad = [dict(x) for x in self.draws]
            bad[-1] = dict(bad[-1], first=first)
            self.write(bad)
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, "Invalid first"):
                automl.load_draws(self.path)

    def test_second_zone_bounds_rejected(self):
        for second in (0, 9):
            bad = [dict(x) for x in self.draws]
            bad[-1] = dict(bad[-1], second=second)
            self.write(bad)
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "Invalid second"):
                automl.load_draws(self.path)


if __name__ == "__main__":
    unittest.main()
