"""Regression tests for V6 archive freshness and exact-selector consistency."""
import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from v6_exact_e2e import validate, run


class V6LatestArchiveContract(unittest.TestCase):
    def test_history_mirrors_match_full_archive(self):
        full = validate(ROOT / "historical_archive.csv")
        self.assertEqual(full[-10:], validate(ROOT / "historical_latest_10.csv"))
        self.assertEqual(full[-20:], validate(ROOT / "historical_latest_20.csv"))

    def test_latest_archive_exact_selection_contract(self):
        path = ROOT / "historical_archive.csv"
        rows = validate(path)
        result = run(path)
        self.assertEqual(result["last_draw"], rows[-1][0])
        self.assertEqual(result["views"]["complete_archive"]["rows"], len(rows))
        self.assertEqual(result["views"]["latest_5"]["to"], rows[-1][0])
        self.assertEqual(result["views"]["latest_10"]["to"], rows[-1][0])
        self.assertEqual(result["exact_optimization_layer"]["searched_combinations"], 28)
        self.assertEqual(result["first_zone"], result["exact_optimization_layer"]["exact_selection"])
        self.assertEqual(len(set(result["first_zone"])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in result["first_zone"]))
        self.assertTrue(1 <= result["second_zone"] <= 8)


if __name__ == "__main__":
    unittest.main()
