import csv
import json
import tempfile
import unittest
import runpy
import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path

from v6_advanced_feature_stack import _minmax, build_features, main as feature_main, run


def make_history(path, count=16):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["draw_id", "first", "second"])
        for i in range(count):
            nums = sorted({((i * 7 + j * 5) % 38) + 1 for j in range(6)})
            writer.writerow([str(115000001 + i), " ".join(f"{n:02d}" for n in nums), (i % 8) + 1])


class AdvancedFeatureStackTests(unittest.TestCase):
    def test_real_tsfresh_featuretools_to_exact_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            make_history(path)
            result = run(path)
        layer = result["advanced_feature_stack"]
        self.assertTrue(layer["executed"])
        self.assertEqual(layer["engine"], "tsfresh+Featuretools")
        self.assertEqual(layer["feature_window"], [5, 10])
        self.assertEqual(layer["current_feature_history_rows"], 10)
        self.assertEqual(layer["candidate_count"], 38)
        self.assertGreater(layer["tsfresh_feature_count"], 0)
        self.assertGreater(layer["featuretools_feature_count"], 0)
        self.assertEqual(len(result["candidate_pool"]), 8)
        self.assertEqual(len(result["first_zone"]), 6)
        self.assertEqual(len(set(result["first_zone"])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in result["first_zone"]))
        self.assertTrue(1 <= result["second_zone"] <= 8)
        self.assertEqual(result["exact_optimization_layer"]["searched_combinations"], 28)
        self.assertFalse(layer["predictive_advantage_verified"])

    def test_requires_ten_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            make_history(path, count=9)
            from v6_exact_e2e import validate
            with self.assertRaisesRegex(ValueError, "at least 10"):
                validate(path)



    def test_minmax_constant_and_variable(self):
        self.assertEqual(_minmax([3.0, 3.0]), [0.0, 0.0])
        self.assertEqual(_minmax([2.0, 4.0]), [0.0, 1.0])

    def test_too_few_history_rows_for_engine(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            build_features([])

    def test_incomplete_and_nonfinite_feature_matrices_rejected(self):
        import pandas as pd
        rows = [(str(i), [1, 2, 3, 4, 5, 6]) for i in range(10)]
        ts = pd.DataFrame({"value__mean": [float(i) / 38 for i in range(38)]},
                          index=pd.Index(range(1, 39), name="id"))
        short = pd.DataFrame({f"col{k}": list(range(37)) for k in range(5)})
        normal = pd.DataFrame({f"col{k}": list(range(38)) for k in range(5)})
        with patch("tsfresh.extract_features", return_value=ts), \
             patch("featuretools.dfs", return_value=(short, [])):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                build_features(rows)
        with patch("tsfresh.extract_features", return_value=ts), \
             patch("featuretools.dfs", return_value=(normal, [])), \
             patch("v6_advanced_feature_stack._minmax", side_effect=lambda values: [float("nan")] * len(values)):
            with self.assertRaisesRegex(RuntimeError, "invalid candidate scores"):
                build_features(rows)

    def test_cli_with_output_and_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "advanced.json"
            with patch("v6_advanced_feature_stack.run", return_value={"test": "ok"}), \
                 patch.object(sys, "argv", ["advanced", "--csv", "fake.csv", "--output", str(output)]), \
                 redirect_stdout(StringIO()):
                feature_main()
            self.assertEqual(output.read_text(encoding="utf-8").strip(), '{\n  "test": "ok"\n}')
            with patch("v6_advanced_feature_stack.run", return_value={"test": "ok"}), \
                 patch.object(sys, "argv", ["advanced", "--csv", "fake.csv"]), \
                 redirect_stdout(StringIO()):
                feature_main()

    def test_script_entrypoint_with_mocked_external_engines(self):
        import pandas as pd
        rows = [(str(i), [1, 2, 3, 4, 5, 6]) for i in range(10)]
        ts = pd.DataFrame({"value__mean": [float(i) for i in range(38)]},
                          index=pd.Index(range(1, 39), name="id"))
        normal = pd.DataFrame({f"col{k}": list(range(38)) for k in range(5)})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "advanced.json"
            with patch("tsfresh.extract_features", return_value=ts), \
                 patch("featuretools.dfs", return_value=(normal, [1, 2, 3, 4, 5])), \
                 patch("v6_exact_e2e.validate", return_value=rows), \
                 patch("v6_exact_e2e.complete_result", return_value={"first_zone": [1, 2, 3, 4, 5, 6]}), \
                 patch.object(sys, "argv", ["v6_advanced_feature_stack", "--csv", "fake.csv", "--output", str(output)]), \
                 redirect_stdout(StringIO()):
                runpy.run_module("v6_advanced_feature_stack", run_name="__main__")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["advanced_feature_stack"]["executed"])

if __name__ == "__main__":
    unittest.main()
