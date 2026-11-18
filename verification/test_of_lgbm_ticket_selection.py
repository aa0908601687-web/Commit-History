"""Standalone OF-LGBM combination/data-validation regression tests."""
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import date, timedelta
import csv
sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_ticket_selection import select_ticket
from of_lgbm_real_history_gate import audit_csv

class TicketSelectionTests(unittest.TestCase):
    def setUp(self):
        self.scores = [0.15] * 38
        for n in range(1, 9):
            self.scores[n - 1] = 0.8
        self.history = [tuple(range(1, 7))] * 20

    def test_deterministic_and_exact_pool_count(self):
        ticket, info = select_ticket(self.scores, self.history, 6, 38)
        again, _ = select_ticket(self.scores, self.history, 6, 38)
        self.assertEqual(ticket, again)
        self.assertEqual(len(ticket), 6)
        self.assertEqual(info["candidate_combinations"], 28)
        self.assertFalse(info["pair_signal_verified_predictive"])

    def test_recent_pairs_change_ticket_when_scores_tied(self):
        ticket1, _ = select_ticket(self.scores, [tuple(range(1, 7))] * 20, 6, 38)
        ticket2, _ = select_ticket(self.scores, [tuple(range(3, 9))] * 20, 6, 38)
        self.assertNotEqual(ticket1, ticket2)

    def test_future_not_read(self):
        t1, _ = select_ticket(self.scores, self.history, 6, 38)
        history2 = [tuple(range(3, 9))] + self.history
        t2, _ = select_ticket(self.scores, history2, 6, 38)
        self.assertEqual(t1, t2)  # earlier than last 20 does not affect scoring

    def test_539_pool_size(self):
        ticket, info = select_ticket([0.5] * 39, [tuple(range(1, 6))] * 20, 5, 39)
        self.assertEqual(len(ticket), 5)
        self.assertEqual(info["candidate_combinations"], 56)

    def test_rejects_nonfinite_scores(self):
        scores = self.scores.copy()
        scores[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            select_ticket(scores, self.history, 6, 38)

class DataGateTests(unittest.TestCase):
    def test_partial_date_column_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["draw_id", "draw_date", "first", "second"])
                for i in range(20):
                    w.writerow([f"115{i+1:06d}", "" if i == 5 else (date(2026, 1, 1)+timedelta(days=i)).isoformat(), "01 02 03 04 05 06", "01"])
            with self.assertRaisesRegex(ValueError, "invalid values"):
                audit_csv(p, "power")

    def test_power_missing_dates_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["draw_id", "first", "second"])
                for i in range(20):
                    w.writerow([f"115{i+1:06d}", "01 02 03 04 05 06", "01"])
            report = audit_csv(p, "power")
            self.assertEqual(report["date_coverage"], 0)
            self.assertFalse(report["csv_dates_consistent"])

if __name__ == "__main__":
    unittest.main()
