import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from v6_model_audit import (
    BASE_RATE, brier, leave_one_recent_draw_out, top_numbers,
    validate_scores, walk_forward,
)

def fake_rows(size=18):
    rows = []
    for i in range(size):
        numbers = [1 + ((i * 6 + k) % 38) for k in range(6)]
        rows.append((f"{115000001+i:09d}", numbers, 1))
    return rows

def scorer(history):
    # Explicitly depend on the history; never access the target row.
    current = set(history[-1][1])
    return ({n: (0.8 if n in current else 0.1) for n in range(1,39)},
            {"generated_features": 2, "used_features": 2})

class ScoreAuditTests(unittest.TestCase):
    def test_invalid_score_keys_nan_and_range(self):
        base = {n: 0.2 for n in range(1, 39)}
        self.assertEqual(len(validate_scores(base)), 38)
        for invalid in (
            {**base, 1: float("nan")},
            {**base, 1: 1.001},
            {k: v for k,v in base.items() if k != 5},
        ):
            with self.assertRaises(ValueError):
                validate_scores(invalid)

    def test_tie_break_and_score_sum(self):
        scores = {n: 0.5 for n in range(1, 39)}
        self.assertEqual(top_numbers(scores, 6), [1,2,3,4,5,6])
        self.assertAlmostEqual(brier(scores, [1,2,3,4,5,6]), 0.25)

    def test_no_target_leakage_in_walkforward(self):
        rows = fake_rows()
        seen = []
        def monitored(history):
            seen.append(tuple(row[0] for row in history))
            return scorer(history)
        report = walk_forward(rows, monitored, min_history=12, max_steps=3)
        self.assertEqual(report["evaluated_draws"], 3)
        self.assertEqual(len(seen), 3)
        for j, ids in enumerate(seen):
            target = rows[-3+j][0]
            self.assertEqual(ids[-1], rows[-4+j][0])
            self.assertNotIn(target, ids)
            self.assertEqual(report["per_draw"][j]["target_draw_id"], target)
        self.assertAlmostEqual(report["fair_baseline_brier"], BASE_RATE*(1-BASE_RATE))

    def test_sensitivity_perturbations(self):
        rows = fake_rows()
        audit = leave_one_recent_draw_out(rows, scorer)
        self.assertEqual(len(audit["trials"]), 10)
        self.assertTrue(all(0 <= t["top6_overlap"] <= 6 for t in audit["trials"]))
        self.assertTrue(all(0 <= t["pool8_overlap"] <= 8 for t in audit["trials"]))
        self.assertEqual(audit["reference_last_draw"], rows[-1][0])

    def test_no_evaluation_possible_without_future_target(self):
        with self.assertRaises(ValueError):
            walk_forward(fake_rows(12), scorer)

if __name__ == "__main__":
    unittest.main()
