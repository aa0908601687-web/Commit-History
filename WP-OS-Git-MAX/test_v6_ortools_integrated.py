"""Integrated CP-SAT verification on real archived V6 rows and first-zone scores."""
import pathlib
import sys
import unittest
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v6_exact_e2e as e2e


class IntegratedOrtoolsTests(unittest.TestCase):
    def test_ortools_enabled_and_disabled(self):
        csv_path = HERE / "historical_archive.csv"
        rows = e2e.validate(csv_path)
        scores = {n: float(n) / 38 for n in range(1, 39)}
        base = e2e.complete_result(csv_path, rows, scores, {"enabled": False})
        self.assertFalse(base["ortools_layer"]["executed"])
        result = e2e.complete_result(csv_path, rows, scores, {"enabled": False}, ortools_check=True)
        self.assertTrue(result["ortools_layer"]["executed"])
        self.assertTrue(result["ortools_layer"]["same_optimal_score"])
        self.assertEqual(result["first_zone"], base["first_zone"])

    def test_solver_disagreement_fails_closed(self):
        csv_path = HERE / "historical_archive.csv"
        rows = e2e.validate(csv_path)
        scores = {n: float(n) / 38 for n in range(1, 39)}
        with patch("v6_five_star_stack.ortools_six", return_value=([1, 2, 3, 4, 5, 6], -1.0)):
            with self.assertRaisesRegex(RuntimeError, "disagrees"):
                e2e.complete_result(csv_path, rows, scores, {"enabled": False}, ortools_check=True)


if __name__ == "__main__":
    unittest.main()
