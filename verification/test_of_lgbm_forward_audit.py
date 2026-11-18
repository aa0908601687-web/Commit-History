"""Isolated OF-LGBM CSV and holdout contracts. No quantum or V6 code."""
from __future__ import annotations
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of_lgbm_forward_audit as audit


def fixture(path, count=20):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["draw_id", "first", "second"])
        for i in range(count):
            writer.writerow([f"115{i+1:06d}", "01 02 03 04 05 06", (i % 8) + 1])


class ForwardAuditTests(unittest.TestCase):
    def test_valid_csv_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            fixture(path)
            ids, draws, seconds, record = audit.audit_history(path, "power")
        self.assertEqual(record["rows"], 20)
        self.assertEqual(record["first_id"], "115000001")
        self.assertEqual(record["last_id"], "115000020")
        self.assertFalse(record["independent_official_provenance_verified"])
        self.assertEqual(len(draws), 20)
        self.assertEqual(len(seconds), 20)

    def test_rejects_gap_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            fixture(path)
            lines = path.read_text().splitlines()
            lines[5] = lines[5].replace("115000005", "115000006")
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "nonconsecutive"):
                audit.audit_history(path, "power")

    def test_rejects_invalid_first_zone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            fixture(path)
            lines = path.read_text().splitlines()
            lines[2] = lines[2].replace("01 02 03 04 05 06", "01 02 03 04 05 39")
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid first.zone"):
                audit.audit_history(path, "power")

    def test_holdout_never_sees_future_draw(self):
        ids = list(range(115000001, 115000031))
        draws = [tuple(range(1, 7))] * 30
        seconds = [(i % 8) + 1 for i in range(30)]
        observed = []
        def stub(prior, prior_seconds, game):
            observed.append((len(prior), len(prior_seconds), game))
            return (tuple(range(1, 7)), 1)
        with patch.object(audit, "predict_one", side_effect=stub):
            result = audit.walk_forward(ids, draws, seconds, "power", folds=2)
        self.assertEqual(observed, [(28, 28, "power"), (29, 29, "power")])
        self.assertEqual([r["target_id"] for r in result["outcomes"]],
                         ["115000029", "115000030"])
        self.assertEqual([r["cutoff_id"] for r in result["outcomes"]],
                         ["115000028", "115000029"])
        self.assertEqual(result["distribution"]["6"], 2)
        self.assertFalse(result["predictive_advantage_verified"])

    def test_null_probability_exact(self):
        p = audit.theory_3plus(38, 6)
        expected = sum(math.comb(6, k) * math.comb(32, 6-k)
                       for k in range(3, 7)) / math.comb(38, 6)
        self.assertAlmostEqual(p, expected)
        self.assertGreater(p, 0)
        self.assertLess(p, 1)


if __name__ == "__main__":
    unittest.main()
