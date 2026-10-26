import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v3cache", ROOT / "three_layer_cache_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

class ThreeLayerCacheV3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "draws.csv"
        source = ROOT / "historical_latest_20.csv"
        self.csv.write_bytes(source.read_bytes())
        self.feature = self.root / "features.json"
        self.prediction = self.root / "prediction.json"

    def run_v3(self):
        return v3.run(self.csv, self.feature, self.prediction, verify=True)

    def test_first_run_then_hit_and_identical_output(self):
        a = self.run_v3()
        self.assertEqual((a["feature_cache"], a["prediction_cache"]), ("refreshed","refreshed"))
        b = self.run_v3()
        self.assertEqual((b["feature_cache"], b["prediction_cache"]), ("hit","hit"))
        self.assertEqual(a["prediction"], b["prediction"])
        self.assertEqual(b["selection_window"], [5,10])

    def test_tampered_result_refreshes(self):
        self.run_v3()
        obj = json.loads(self.prediction.read_text())
        obj["prediction"]["first"][0] = 38
        self.prediction.write_text(json.dumps(obj))
        r = self.run_v3()
        self.assertEqual(r["prediction_cache"], "refreshed")

    def test_corrupted_json_refreshes(self):
        self.run_v3()
        self.feature.write_text("{broken")
        self.prediction.write_text("{broken")
        r = self.run_v3()
        self.assertEqual(r["feature_cache"], "refreshed")
        self.assertEqual(r["prediction_cache"], "refreshed")

    def test_changed_recent_draw_invalidates_both(self):
        self.run_v3()
        rows = self.csv.read_text().splitlines()
        last = rows[-1].split(",")
        last[1] = "01 09 10 21 22 32"
        rows[-1] = ",".join(last)
        self.csv.write_text("\n".join(rows)+"\n")
        r = self.run_v3()
        self.assertEqual(r["feature_cache"], "refreshed")
        self.assertEqual(r["prediction_cache"], "refreshed")

    def test_schema_change_invalidates_result(self):
        self.run_v3()
        obj = json.loads(self.prediction.read_text())
        obj["cache_schema_version"] = 999
        obj["sha256"] = v3.digest({k:v for k,v in obj.items() if k != "sha256"})
        self.prediction.write_text(json.dumps(obj))
        self.assertEqual(self.run_v3()["prediction_cache"], "refreshed")

    def test_prediction_source_mismatch_rejected(self):
        self.run_v3()
        with mock.patch.object(v3, "predict", return_value={"through_draw": "000000000"}):
            with self.assertRaisesRegex(ValueError, "Prediction source mismatch"):
                v3.run(self.csv, self.feature, self.root / "new_prediction.json")

    def test_prediction_cache_invalid_payload_refreshes(self):
        self.run_v3()
        obj = json.loads(self.prediction.read_text())
        obj["prediction"]["through_draw"] = "000000000"
        obj["sha256"] = v3.digest({k: v for k, v in obj.items() if k != "sha256"})
        self.prediction.write_text(json.dumps(obj))
        self.assertEqual(self.run_v3()["prediction_cache"], "refreshed")

    def test_verify_detects_divergence(self):
        self.run_v3()
        with mock.patch.object(v3, "predict", return_value={"through_draw": "000000000"}):
            with self.assertRaisesRegex(ValueError, "Cached and direct predictions differ"):
                v3.run(self.csv, self.feature, self.prediction, verify=True)

    def test_short_history_is_rejected(self):
        self.csv.write_text(chr(10).join(self.csv.read_text().splitlines()[:10]) + chr(10))
        with self.assertRaisesRegex(ValueError, "At least ten prior draws"):
            self.run_v3()

    def test_cli_main_outputs_json(self):
        args = ["cache_v3", "--csv", str(self.csv), "--feature-cache",
                str(self.feature), "--prediction-cache", str(self.prediction), "--verify"]
        buf = StringIO()
        with mock.patch("sys.argv", args), redirect_stdout(buf):
            v3.main()
        result = json.loads(buf.getvalue())
        self.assertTrue(result["consistency_verified"])
        self.assertEqual(result["selection_window"], [5, 10])

if __name__ == "__main__":
    unittest.main()
