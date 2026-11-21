import csv
import tempfile
import unittest
from pathlib import Path

from research_runner import run_research_csv


class ResearchRunnerTests(unittest.TestCase):
    def test_csv_to_model_to_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["draw_id", "date", "n1", "n2", "n3", "n4", "n5", "n6", "second"])
                for n in range(1, 23):
                    writer.writerow([str(n), f"2026-01-{n:02d}", 1, 2, 3, 4, 5, 6, 1])
            calls = []

            def fake_trainer(train_x, train_y, valid_x, valid_y, *, seed, n_jobs):
                calls.append((len(train_x), len(valid_x), len(train_y)))
                probabilities = [float(n) / len(valid_x) for n in valid_x["number"]]
                return {"probabilities": [[1 - p, p] for p in probabilities]}

            result = run_research_csv(path, trainer=fake_trainer)
            self.assertEqual(result["history_draws"], 22)
            self.assertEqual(result["last_draw_id"], "22")
            self.assertEqual(result["selection"]["first_area"], tuple(range(33, 39)))
            self.assertEqual(result["selection"]["second_area"], 8)
            self.assertEqual(calls, [(21 * 38, 38, 21 * 38), (21 * 8, 8, 21 * 8)])
            self.assertEqual(len(result["source_sha256"]), 64)

    def test_reject_insufficient_history(self):
        with self.assertRaises(ValueError):
            run_research_csv("unused.csv", min_history=1)


if __name__ == "__main__":
    unittest.main()
