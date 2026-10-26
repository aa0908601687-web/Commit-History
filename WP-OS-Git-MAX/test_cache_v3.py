"""Regression checks: V3 cache must never change original selection output."""
import csv
import contextlib
import io
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp_os_git_max import load, predict
from unittest.mock import patch
import cache_v3
from cache_v3 import run

ROOT = Path(__file__).resolve().parent

class V3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.csv = ROOT / "historical_latest_20.csv"

    def test_cached_vs_direct_and_reuse(self):
        direct = predict(load(self.csv))
        first = run(self.csv, self.folder)
        second = run(self.csv, self.folder)
        self.assertEqual(first["prediction"], direct)
        self.assertEqual(second["prediction"], direct)
        self.assertEqual(first["cache"]["prediction"], "refreshed")
        self.assertEqual(second["cache"], {"draws":"hit","features":"hit","prediction":"hit"})

    def test_tampering_refreshes(self):
        run(self.csv, self.folder)
        p = self.folder / "prediction.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["prediction"]["first"] = [1, 2, 3, 4, 5, 6]
        p.write_text(json.dumps(obj), encoding="utf-8")
        result = run(self.csv, self.folder)
        self.assertEqual(result["cache"]["prediction"], "refreshed")
        self.assertEqual(result["prediction"], predict(load(self.csv)))

    def test_changed_history_refreshes_layers(self):
        run(self.csv, self.folder)
        with self.csv.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        rows[-1]["second"] = "01" if rows[-1]["second"] != "01" else "02"
        alternative = self.folder / "changed.csv"
        with alternative.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["draw_id","first","second"])
            writer.writeheader()
            writer.writerows(rows)
        result = run(alternative, self.folder)
        self.assertEqual(result["cache"], {"draws":"refreshed","features":"refreshed","prediction":"refreshed"})
        self.assertEqual(result["prediction"], predict(load(alternative)))

    def test_modified_prediction_model_version_refreshes(self):
        run(self.csv, self.folder)
        p = self.folder / "prediction.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["model_version"] = "old"
        p.write_text(json.dumps(obj), encoding="utf-8")
        self.assertEqual(run(self.csv, self.folder)["cache"]["prediction"], "refreshed")

    def test_bad_draw_cache_rebuilds(self):
        run(self.csv, self.folder)
        (self.folder / "draws.json").write_text("{invalid", encoding="utf-8")
        result = run(self.csv, self.folder)
        self.assertEqual(result["cache"]["draws"], "refreshed")
        self.assertEqual(result["cache"]["prediction"], "hit")

    def test_bad_prediction_cache_rebuilds(self):
        run(self.csv, self.folder)
        (self.folder / "prediction.json").write_text("{invalid", encoding="utf-8")
        result = run(self.csv, self.folder)
        self.assertEqual(result["cache"]["prediction"], "refreshed")
        self.assertEqual(result["prediction"], predict(load(self.csv)))

    def test_cli_outputs_prediction(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["cache_v3.py", "--csv", str(self.csv), "--cache-dir", str(self.folder)]):
            with contextlib.redirect_stdout(output):
                cache_v3.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["prediction"]["first"], predict(load(self.csv))["first"])
        self.assertEqual(result["cache"]["prediction"], "refreshed")

    def test_rejects_insufficient_draws(self):
        with self.csv.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        short = self.folder / "short.csv"
        with short.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["draw_id", "first", "second"])
            writer.writeheader()
            writer.writerows(rows[:9])
        with self.assertRaisesRegex(ValueError, "Need at least 10 draws"):
            run(short, self.folder)

    def test_main_guard_in_coverage_process(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["cache_v3.py", "--csv", str(self.csv),
                                        "--cache-dir", str(self.folder)]):
            with contextlib.redirect_stdout(output):
                runpy.run_path(str(ROOT / "cache_v3.py"), run_name="__main__")
        self.assertEqual(json.loads(output.getvalue())["prediction"]["first"],
                         predict(load(self.csv))["first"])

    def test_actual_script_entrypoint(self):
        script = ROOT / "cache_v3.py"
        result = subprocess.run(
            [sys.executable, str(script), "--csv", str(self.csv),
             "--cache-dir", str(self.folder)], capture_output=True,
            text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["prediction"]["first"],
                         predict(load(self.csv))["first"])

if __name__ == "__main__":
    unittest.main()
