import contextlib
import csv
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cache_v3_pipeline as v3


class CacheV3CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / "draws.csv"
        self.cache = self.root / "cache.json"
        self.draws = [
            {"draw_id": str(115000001+i),
             "first": sorted(((i+j*5) % 38)+1 for j in range(6)),
             "second": i%8+1}
            for i in range(12)
        ]
        with self.csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(("draw_id", "first", "second"))
            for d in self.draws:
                writer.writerow((d["draw_id"],
                                 " ".join(map(str, d["first"])), d["second"]))

    def invoke(self, *args, as_script=False):
        argv = ["cache_v3_pipeline.py", "--csv", str(self.csv), "--cache", str(self.cache), *args]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            if as_script:
                runpy.run_module("cache_v3_pipeline", run_name="__main__")
            else:
                v3.main()
        return json.loads(output.getvalue())

    def test_cli_cache_refresh_hit_and_prediction(self):
        pred = self.root / "prediction.json"
        first = self.invoke("--prediction", str(pred), as_script=True)
        self.assertEqual(first["cache_status"], "refreshed")
        self.assertEqual(first["schema"], 3)
        self.assertEqual(json.loads(pred.read_text())["first"], first["prediction"]["first"])
        same = self.invoke()
        self.assertEqual(same["cache_status"], "hit")

    def test_cli_snapshot_and_validation(self):
        original_csv = self.csv.read_text()
        lines = original_csv.splitlines()
        self.csv.write_text("\n".join(lines[:-1]) + "\n")
        target = self.draws[-1]["draw_id"]
        snap = self.root / "snapshot.json"
        self.invoke("--snapshot", str(snap), "--target-draw", target)
        self.assertTrue(snap.exists())
        self.csv.write_text(original_csv)
        result = self.invoke("--validate-snapshot", str(snap))
        self.assertEqual(result["validation"]["target_draw"], target)

    def test_cli_missing_target_and_no_overwrite(self):
        p = self.root / "snap.json"
        with self.assertRaises(SystemExit):
            self.invoke("--snapshot", str(p))
        self.invoke("--snapshot", str(p), "--target-draw", "115000013")
        with self.assertRaises(FileExistsError):
            self.invoke("--snapshot", str(p), "--target-draw", "115000013")

    def test_cli_fails_closed_for_drift(self):
        with patch.object(v3, "legacy_predict", return_value={"different": True}):
            with self.assertRaisesRegex(ValueError, "differs"):
                self.invoke()

    def test_invalid_target_short_history_and_corrupt_cache(self):
        with self.assertRaises(ValueError):
            v3.expected(self.draws[:9])
        prediction=v3.predict_from_cache(v3.expected(self.draws))
        for bad in ("wrong", "115000012"):
            with self.assertRaises(ValueError):
                v3.snapshot(self.draws, prediction, bad)
        self.cache.write_text("{invalid JSON", encoding="utf-8")
        self.assertEqual(self.invoke()["cache_status"], "refreshed")
