"""Guard the latest-draw CSV windows against accidental full-archive copies."""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


class DrawWindowConsistencyTests(unittest.TestCase):
    def test_latest_windows_match_archive_tail(self):
        archive = load("historical_archive.csv")
        recent20 = load("historical_latest_20.csv")
        recent10 = load("historical_latest_10.csv")
        self.assertGreaterEqual(len(archive), 20)
        self.assertEqual(len(recent20), 20)
        self.assertEqual(len(recent10), 10)
        self.assertEqual(recent20, archive[-20:])
        self.assertEqual(recent10, archive[-10:])
        ids = [int(row["draw_id"]) for row in archive]
        self.assertEqual(ids, sorted(set(ids)))


if __name__ == "__main__":
    unittest.main()
