import csv
import tempfile
import unittest
from pathlib import Path

from history_adapter import load_verified_format, historical_windows, FIELDS


class HistoryAdapterTests(unittest.TestCase):
    def _csv(self, rows):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "draws.csv"
        with path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(tmp.cleanup)
        return path

    def _row(self, idx):
        return {"draw_id": str(idx), "date": f"2026-01-{idx:02d}",
                **{f"n{i}": i for i in range(1, 7)}, "second": 1}

    def test_strict_parse_and_no_future_leakage(self):
        rows = load_verified_format(self._csv([self._row(i) for i in range(1, 23)]))
        views = historical_windows(rows, 21)
        self.assertEqual(len(views["last_10"]), 10)
        self.assertEqual(len(views["last_20"]), 20)
        self.assertEqual(len(views["all_history"]), 21)
        self.assertEqual(views["all_history"][-1]["draw_id"], "21")

    def test_duplicate_draw_rejected(self):
        with self.assertRaises(ValueError):
            load_verified_format(self._csv([self._row(1), self._row(1)]))

    def test_out_of_order_rejected(self):
        with self.assertRaises(ValueError):
            load_verified_format(self._csv([self._row(2), self._row(1)]))

    def test_invalid_number_rejected(self):
        row = self._row(1)
        row["n6"] = 39
        with self.assertRaises(ValueError):
            load_verified_format(self._csv([row]))

    def test_no_prior_draw_rejected(self):
        with self.assertRaises(ValueError):
            historical_windows([self._row(1)], 0)


if __name__ == "__main__":
    unittest.main()
