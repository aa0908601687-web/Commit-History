"""Regression gates: no selection feature may read beyond last ten prior draws."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_openfe_bridge import _base_row, VIEW_LIMITS

class FiveTenContractTests(unittest.TestCase):
    def test_views_and_full_archive_context(self):
        self.assertEqual(list(VIEW_LIMITS), ["latest_5", "latest_10", "full_archive"])
        self.assertEqual(VIEW_LIMITS["latest_5"], 5)
        self.assertEqual(VIEW_LIMITS["latest_10"], 10)

    def test_older_draws_cannot_change_current_features(self):
        tail = [(f"1150000{i:02d}", [1,2,3,4,5,6], 1) for i in range(11,21)]
        old_a = [(f"1150000{i:02d}", [7,8,9,10,11,12], 2) for i in range(1,11)]
        old_b = [(f"1150000{i:02d}", [1,3,5,7,9,11], 3) for i in range(1,11)]
        for view_limit in VIEW_LIMITS.values():
            for number in (1, 6, 7, 38):
                self.assertEqual(
                    _base_row(old_a + tail, number, view_limit),
                    _base_row(old_b + tail, number, view_limit),
                )

    def test_full_archive_features_are_ten_draw_capped(self):
        rows = [(f"1150000{i:02d}", [1,2,3,4,5,6], 1) for i in range(1,26)]
        full = _base_row(rows, 1, None)
        self.assertEqual(full["view_rows"], 10.0)
        self.assertEqual(full["gap_view"], 1.0)
        self.assertEqual(_base_row(rows, 1, 5)["view_rows"], 5.0)
        self.assertEqual(_base_row(rows, 1, 10)["view_rows"], 10.0)

if __name__ == "__main__":
    unittest.main()
