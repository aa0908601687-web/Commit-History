"""Fast unit contracts for the opt-in combined CPU E2E; real runtime runs in CI."""
import unittest
import json
import math
import runpy
import sys
import tempfile
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout
from unittest.mock import patch
from v6_full_stack_e2e import fuse_scores, normalize

class FullStackContractTests(unittest.TestCase):
    def test_normalize_and_fuse(self):
        a = {n: float(n) for n in range(1,39)}
        b = {n: float(39-n) for n in range(1,39)}
        scores = fuse_scores(a,b,.5)
        self.assertEqual(set(scores), set(range(1,39)))
        self.assertTrue(all(abs(v - .5) < 1e-9 for v in scores.values()))
    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            normalize({1: 1})
        with self.assertRaises(ValueError):
            fuse_scores({n: n for n in range(1,39)}, {n: n for n in range(1,39)},1.5)
    def test_constant_feature_is_finite(self):
        scores = normalize({n: 4.2 for n in range(1,39)})
        self.assertEqual(set(scores.values()), {0.0})


    def test_nonfinite_score_rejected(self):
        values = {n: float(n) for n in range(1, 39)}
        values[7] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize(values)

    def test_single_invocation_cli_with_real_stage_contracts_mocked(self):
        candidates = {n: float(n) for n in range(1, 39)}
        observed = {
            "last_draw": "115000091",
            "first_zone": [1, 2, 3, 4, 5, 6],
            "second_zone": 1,
            "cudaq_layer": {"executed": True},
            "ising_qubo_layer": {"executed": True},
            "ortools_layer": {"executed": True},
            "exact_optimization_layer": {"searched_combinations": 28},
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            with patch("v6_exact_e2e.validate", return_value=[("test", [1, 2, 3, 4, 5, 6])]), \
                 patch("v6_exact_e2e.complete_result", return_value=observed), \
                 patch("v6_openfe_bridge.score_three_history_views",
                       return_value=(candidates, {"engine": "OpenFE+LightGBM"})), \
                 patch("v6_advanced_feature_stack.build_features",
                       return_value=(candidates, {"executed": True, "engine": "tsfresh+Featuretools"})), \
                 patch.object(sys, "argv", ["v6_full_stack_e2e.py", "--csv", "mock.csv", "--output", str(output)]), \
                 redirect_stdout(StringIO()):
                runpy.run_module("v6_full_stack_e2e", run_name="__main__")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["full_stack_integration"]["executed"])
            self.assertFalse(result["full_stack_integration"]["production_replacement"])
            self.assertEqual(result["exact_optimization_layer"]["searched_combinations"], 28)

if __name__ == "__main__":
    unittest.main()
