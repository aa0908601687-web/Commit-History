import csv
import tempfile
import unittest
from pathlib import Path

from v6_advanced_feature_stack import build_features, run


def make_history(path, count=16):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["draw_id", "first", "second"])
        for i in range(count):
            nums = sorted({((i * 7 + j * 5) % 38) + 1 for j in range(6)})
            writer.writerow([str(115000001 + i), " ".join(f"{n:02d}" for n in nums), (i % 8) + 1])


class AdvancedFeatureStackTests(unittest.TestCase):
    def test_real_tsfresh_featuretools_to_exact_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            make_history(path)
            result = run(path)
        layer = result["advanced_feature_stack"]
        self.assertTrue(layer["executed"])
        self.assertEqual(layer["engine"], "tsfresh+Featuretools")
        self.assertEqual(layer["feature_window"], [5, 10])
        self.assertEqual(layer["current_feature_history_rows"], 10)
        self.assertEqual(layer["candidate_count"], 38)
        self.assertGreater(layer["tsfresh_feature_count"], 0)
        self.assertGreater(layer["featuretools_feature_count"], 0)
        self.assertEqual(len(result["candidate_pool"]), 8)
        self.assertEqual(len(result["first_zone"]), 6)
        self.assertEqual(len(set(result["first_zone"])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in result["first_zone"]))
        self.assertTrue(1 <= result["second_zone"] <= 8)
        self.assertEqual(result["exact_optimization_layer"]["searched_combinations"], 28)
        self.assertFalse(layer["predictive_advantage_verified"])

    def test_requires_ten_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            make_history(path, count=9)
            from v6_exact_e2e import validate
            rows = validate(path)
            with self.assertRaisesRegex(ValueError, "at least 10"):
                build_features(rows)


if __name__ == "__main__":
    unittest.main()
