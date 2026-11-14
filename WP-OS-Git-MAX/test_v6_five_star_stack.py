"""Coverage and contract tests for the optional five-star stack."""
import contextlib
import importlib.util
import io
import json
import pathlib
import runpy
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import v6_five_star_stack as stack


class FiveStarStackTests(unittest.TestCase):
    def test_availability(self):
        available = stack.availability()
        self.assertEqual(set(available), set(stack.LIBRARIES))
        self.assertTrue(all(isinstance(v, bool) for v in available.values()))

    def test_exact_six_and_input_contracts(self):
        pool = [9, 15, 7, 27, 18, 19, 32, 24]
        scores = {n: float(39 - n) / 38 for n in pool}
        result, value, count = stack.exact_six(pool, scores)
        self.assertEqual(count, 28)
        self.assertEqual(result, [7, 9, 15, 18, 19, 24])
        self.assertAlmostEqual(value, sum(scores[n] for n in result))
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            stack.exact_six(pool[:-1], scores)
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            stack.exact_six([9] * 8, {9: 1.0})
        with self.assertRaisesRegex(ValueError, "out of range"):
            stack.exact_six([0, 15, 7, 27, 18, 19, 32, 24], scores)
        with self.assertRaisesRegex(ValueError, "scores must cover"):
            stack.exact_six(pool, {9: 1.0})

    def test_run_without_optional_solver(self):
        with patch.object(stack, "availability", return_value={"ortools": False}):
            result = stack.run()
        self.assertFalse(result["production_model_executed"])
        self.assertEqual(result["exact_reference"]["combinations"], 28)
        self.assertNotIn("ortools", result)

    def test_ortools_solver_and_run(self):
        try:
            import ortools  # noqa: F401
        except ImportError:
            self.skipTest("OR-Tools not installed")
        pool = [9, 15, 7, 27, 18, 19, 32, 24]
        scores = {n: float(39 - n) / 38 for n in pool}
        selected, value = stack.ortools_six(pool, scores)
        reference, expected, _ = stack.exact_six(pool, scores)
        self.assertEqual(selected, reference)
        self.assertAlmostEqual(value, expected)
        result = stack.run()
        self.assertTrue(result["ortools"]["matches_reference_score"])

    def test_ortools_not_optimal(self):
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            self.skipTest("OR-Tools not installed")
        with patch.object(cp_model.CpSolver, "solve", return_value=cp_model.UNKNOWN):
            with self.assertRaisesRegex(RuntimeError, "did not certify"):
                stack.ortools_six([1, 2, 3, 4, 5, 6, 7, 8],
                                 {n: float(n) for n in range(1, 9)})

    def test_command_line(self):
        # runpy exercises the __main__ path in the coverage job.
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(ROOT / "v6_five_star_stack.py"), run_name="__main__")
        data = json.loads(output.getvalue())
        self.assertEqual(data["exact_reference"]["combinations"], 28)


if __name__ == "__main__":
    unittest.main()
