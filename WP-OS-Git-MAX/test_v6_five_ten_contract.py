"""Regression gates for separate 10/20/all-prior OpenFE feature views."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_openfe_bridge import _base_row, VIEW_LIMITS

class ThreeViewContractTests(unittest.TestCase):
    def test_views_and_full_archive_context(self):
        self.assertEqual(list(VIEW_LIMITS), ["latest_10", "latest_20", "full_archive"])
        self.assertEqual(VIEW_LIMITS["latest_10"], 10)
        self.assertEqual(VIEW_LIMITS["latest_20"], 20)
        self.assertIsNone(VIEW_LIMITS["full_archive"])

    def test_recent_window_ignores_older_draws(self):
        tail = [(f"1150000{i:02d}", [1,2,3,4,5,6], 1) for i in range(21,41)]
        old_a = [(f"1150000{i:02d}", [7,8,9,10,11,12], 2) for i in range(1,21)]
        old_b = [(f"1150000{i:02d}", [1,3,5,7,9,11], 3) for i in range(1,21)]
        for view_limit in (10, 20):
            for number in (1, 6, 7, 38):
                self.assertEqual(
                    _base_row(old_a + tail, number, view_limit),
                    _base_row(old_b + tail, number, view_limit),
                )

    def test_full_archive_changes_when_older_draws_change(self):
        tail = [(f"1150000{i:02d}", [1,2,3,4,5,6], 1) for i in range(21,41)]
        old_a = [(f"1150000{i:02d}", [7,8,9,10,11,12], 2) for i in range(1,21)]
        old_b = [(f"1150000{i:02d}", [1,3,5,7,9,11], 3) for i in range(1,21)]
        a = _base_row(old_a + tail, 1, None)
        b = _base_row(old_b + tail, 1, None)
        self.assertEqual(a["view_rows"], 40.0)
        self.assertNotEqual(a["freq_view"], b["freq_view"])

    def test_view_window_sizes(self):
        rows = [(f"1150000{i:02d}", [1,2,3,4,5,6], 1) for i in range(1,26)]
        self.assertEqual(_base_row(rows, 1, None)["view_rows"], 25.0)
        self.assertEqual(_base_row(rows, 1, 10)["view_rows"], 10.0)
        self.assertEqual(_base_row(rows, 1, 20)["view_rows"], 20.0)

if __name__ == "__main__":
    unittest.main()
