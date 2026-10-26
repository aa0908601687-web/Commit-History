import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import research_cache_v3 as cache


class ResearchCacheV3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "draws.csv"
        source = Path(__file__).resolve().parent / "historical_latest_20.csv"
        self.csv.write_bytes(source.read_bytes())
        self.control = self.root / "control.json"
        self.fusion = self.root / "fusion.json"

    def run_cache(self):
        return cache.run(self.csv, self.control, self.fusion, verify=True)

    def test_refresh_then_hit_and_direct_parity(self):
        first = self.run_cache()
        self.assertEqual(first["models"]["control_10_20"]["cache"], "refreshed")
        self.assertEqual(first["models"]["fusion_5_10_20"]["cache"], "refreshed")
        second = self.run_cache()
        self.assertEqual(second["models"]["control_10_20"]["cache"], "hit")
        self.assertEqual(second["models"]["fusion_5_10_20"]["cache"], "hit")
        self.assertTrue(second["consistency_verified"])

    def test_tampered_and_corrupt_caches_refresh(self):
        self.run_cache()
        obj = json.loads(self.control.read_text())
        obj["prediction"]["first"][0] = 38
        self.control.write_text(json.dumps(obj))
        self.fusion.write_text("{broken")
        result = self.run_cache()
        self.assertEqual(result["models"]["control_10_20"]["cache"], "refreshed")
        self.assertEqual(result["models"]["fusion_5_10_20"]["cache"], "refreshed")

    def test_intact_wrapper_with_invalid_prediction_refreshes(self):
        self.run_cache()
        obj = json.loads(self.control.read_text())
        obj["prediction"]["through_draw"] = "000000000"
        obj["sha256"] = cache.digest({k: v for k, v in obj.items() if k != "sha256"})
        self.control.write_text(json.dumps(obj))
        result = self.run_cache()
        self.assertEqual(result["models"]["control_10_20"]["cache"], "refreshed")

    def test_intact_but_stale_key_refreshes(self):
        self.run_cache()
        obj = json.loads(self.control.read_text())
        obj["model_version"] = "stale"
        obj["sha256"] = cache.digest({k: v for k, v in obj.items() if k != "sha256"})
        self.control.write_text(json.dumps(obj))
        result = self.run_cache()
        self.assertEqual(result["models"]["control_10_20"]["cache"], "refreshed")

    def test_short_history_rejected(self):
        rows = self.csv.read_text().splitlines()
        self.csv.write_text("\n".join(rows[:20]) + "\n")
        with self.assertRaisesRegex(ValueError, "twenty"):
            self.run_cache()

    def test_invalid_new_prediction_rejected(self):
        with mock.patch.dict(cache.MODELS, {
            "control_10_20": ("bad", lambda draws: {"through_draw": "bad"}),
            "fusion_5_10_20": cache.MODELS["fusion_5_10_20"],
        }):
            with self.assertRaisesRegex(ValueError, "source or digest"):
                self.run_cache()

    def test_verify_detects_divergence(self):
        self.run_cache()
        real = cache.MODELS["control_10_20"]
        calls = [real[1](cache.load_draws(self.csv)), {"through_draw": "bad"}]
        with mock.patch.dict(cache.MODELS, {
            "control_10_20": (real[0], mock.Mock(side_effect=calls)),
            "fusion_5_10_20": cache.MODELS["fusion_5_10_20"],
        }):
            with self.assertRaisesRegex(ValueError, "Cached and direct"):
                cache.run(self.csv, self.root / "new.json", self.fusion, verify=True)

    def test_invalid_prediction_shapes(self):
        self.assertFalse(cache._valid_prediction(None, "1"))
        self.assertFalse(cache._valid_prediction({"through_draw": "2"}, "1"))


if __name__ == "__main__":
    unittest.main()
