"""Independent validation checks; existing model and historical data remain untouched."""
import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("wp_os_git_max.py")
spec = importlib.util.spec_from_file_location("wp_os_git_max_validation", MODULE)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class DataAndCalculationValidation(unittest.TestCase):
    def write_csv(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        target = Path(directory.name) / "draws.csv"
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("draw_id", "first", "second"))
            writer.writerows(rows)
        return target

    def example(self):
        return [(str(115000001 + i), "01 04 09 16 23 38", str(i % 8 + 1).zfill(2)) for i in range(10)]

    def test_valid_input_and_sorted_numbers(self):
        draws = model.load(self.write_csv(self.example()))
        self.assertEqual(len(draws), 10)
        self.assertEqual(draws[0]["first"], (1, 4, 9, 16, 23, 38))

    def test_duplicate_period_rejected(self):
        rows = self.example()
        rows[-1] = (rows[0][0], rows[-1][1], rows[-1][2])
        with self.assertRaises(ValueError):
            model.load(self.write_csv(rows))

    def test_invalid_first_zone_rejected(self):
        for first in ("01 01 03 04 05 06", "00 02 03 04 05 06",
                      "01 02 03 04 05 39", "01 02 03 04 05"):
            with self.subTest(first=first), self.assertRaises(ValueError):
                model.load(self.write_csv([("115000001", first, "01")]))

    def test_invalid_second_zone_rejected(self):
        for second in ("00", "09"):
            with self.subTest(second=second), self.assertRaises(ValueError):
                model.load(self.write_csv([("115000001", "01 02 03 04 05 06", second)]))

    def test_period_format_and_order_rejected(self):
        for bad in ("123", "11500000A"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.load(self.write_csv([(bad, "01 02 03 04 05 06", "01")]))
        rows = self.example()[::-1]
        with self.assertRaises(ValueError):
            model.load(self.write_csv(rows))

    def test_exact_weights_and_single_zone_counts(self):
        draws = model.load(self.write_csv(self.example()))
        ranking = dict(model.rank_draws([d["first"] for d in draws], 38))
        n = 1
        c5 = 5
        c10 = 10
        age = 0
        expected = (.45*c5/5 + .25*c10/10 + .15*(c10+1)/(10+38)
                    + .10 + .05*age/10)
        self.assertAlmostEqual(ranking[n], expected, places=11)
        second_ranking = model.score([d["second"] for d in draws], 8)
        self.assertEqual(len(second_ranking), 8)

    def test_prediction_reproducible_and_hash_consistent(self):
        draws = model.load(self.write_csv(self.example()))
        pred = model.predict(draws)
        self.assertEqual(pred, model.predict(draws))
        digest = pred.pop("sha256")
        canonical = json.dumps(pred, sort_keys=True, ensure_ascii=False).encode()
        self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())
        self.assertEqual(len(pred["first"]), 6)
        self.assertEqual(len(set(pred["first"])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in pred["first"]))
        self.assertTrue(1 <= pred["second"] <= 8)

    def test_backtest_no_leakage_and_theoretical_baseline(self):
        draws = model.load(self.write_csv(self.example()))
        result = model.backtest(draws)
        self.assertEqual(result["tested"], 0)
        self.assertEqual(sum(result["first_zone_hits"].values()), 0)
        self.assertEqual(result["fair_jackpot_probability"], 1/(math.comb(38, 6)*8))
        self.assertTrue(0 < result["random_first_zone_3plus"] < 1)


if __name__ == "__main__":
    unittest.main()
