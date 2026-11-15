"""Tests for OF-LGBM extended standalone retrospective evaluation."""
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of_lgbm_extended_walkforward as extended


class ExtendedAuditTests(unittest.TestCase):
    def test_summary_is_explicit_about_insufficient_folds(self):
        rows = [{"target_id": str(i), "matches": 0, "random_matches": 1}
                for i in range(100)]
        result = extended.summarize(rows, "lotto")
        self.assertEqual(result["100"]["status"], "completed")
        self.assertEqual(result["200"]["status"], "not_completed")
        self.assertEqual(result["500"]["status"], "not_completed")
        self.assertEqual(result["100"]["hit_distribution"]["0"], 100)
        self.assertEqual(result["100"]["hits_3plus"], 0)

    def test_exact_fair_tail(self):
        self.assertAlmostEqual(extended.null_tail(0, 100, 0.02), 1.0)
        self.assertAlmostEqual(extended.null_tail(100, 100, 0.02), 0.02 ** 100)
        self.assertLess(extended.null_tail(2, 100, 0.02), 1.0)

    def test_truncated_smoke_never_claims_horizon_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_file = Path(folder) / "539.csv"
            with csv_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["draw_id", "draw_date", "first"])
                # Use valid sequential IDs and actual calendar dates; source is mock only.
                from datetime import date, timedelta
                for i in range(40):
                    writer.writerow([f"115{i+1:06d}",
                                     (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                                     "01 02 03 04 05"])
            with patch.object(extended, "predict_one", return_value=((1, 2, 3, 4, 5), None)):
                result = extended.run("539", csv_file, (100, 200, 500), max_folds=1)
            self.assertEqual(result["executed_folds"], 1)
            self.assertFalse(result["csv_official_provenance_verified"])
            self.assertFalse(result["predictive_advantage_verified"])
            self.assertEqual(result["horizons"]["100"]["status"], "not_completed")
            self.assertEqual(result["outcomes"][0]["cutoff_id"], "115000039")
            self.assertEqual(result["outcomes"][0]["target_id"], "115000040")

    def test_lotto_500_impossible_with_current_411_rows(self):
        self.assertLess(411 - extended.MIN_HISTORY, 500)


if __name__ == "__main__":
    unittest.main()
