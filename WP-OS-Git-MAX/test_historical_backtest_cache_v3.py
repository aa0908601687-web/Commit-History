import json
import tempfile
import unittest
from pathlib import Path

import historical_backtest_cache_v3 as cache


class HistoricalBacktestCacheV3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "draws.csv"
        rows = ["draw_id,first,second"]
        for index in range(25):
            start = index % 33 + 1
            first = " ".join(f"{n:02d}" for n in range(start, start + 6))
            rows.append(f"115000{index + 1:03d},{first},{index % 8 + 1:02d}")
        self.csv.write_text("\n".join(rows) + "\n")
        self.path = self.root / "history.json"

    def test_refresh_then_hit_with_all_models(self):
        first = cache.run(self.csv, self.path)
        self.assertEqual(first["cache"], "refreshed")
        self.assertEqual(set(first["backtests"]), set(cache.MODEL_VERSIONS))
        self.assertEqual(first["backtests"]["main_5_10"]["tested"], 15)
        self.assertEqual(first["backtests"]["control_10_20"]["tested"], 5)
        self.assertEqual(first["backtests"]["fusion_5_10_20"]["tested"], 5)
        second = cache.run(self.csv, self.path)
        self.assertEqual(second["cache"], "hit")
        self.assertEqual(first["backtests"], second["backtests"])

    def test_corrupt_json_refreshes(self):
        self.path.write_text("{broken")
        self.assertEqual(cache.run(self.csv, self.path)["cache"], "refreshed")

    def test_tamper_refreshes(self):
        cache.run(self.csv, self.path)
        obj = json.loads(self.path.read_text())
        obj["backtests"]["main_5_10"]["tested"] = 999
        self.path.write_text(json.dumps(obj))
        self.assertEqual(cache.run(self.csv, self.path)["cache"], "refreshed")

    def test_intact_stale_key_refreshes(self):
        cache.run(self.csv, self.path)
        obj = json.loads(self.path.read_text())
        obj["draw_count"] = 999
        obj["sha256"] = cache.digest({k: v for k, v in obj.items() if k != "sha256"})
        self.path.write_text(json.dumps(obj))
        self.assertEqual(cache.run(self.csv, self.path)["cache"], "refreshed")

    def test_intact_invalid_backtest_shape_refreshes(self):
        cache.run(self.csv, self.path)
        obj = json.loads(self.path.read_text())
        obj["backtests"] = []
        obj["sha256"] = cache.digest({k: v for k, v in obj.items() if k != "sha256"})
        self.path.write_text(json.dumps(obj))
        self.assertEqual(cache.run(self.csv, self.path)["cache"], "refreshed")

    def test_intact_incomplete_models_refreshes(self):
        cache.run(self.csv, self.path)
        obj = json.loads(self.path.read_text())
        del obj["backtests"]["fusion_5_10_20"]
        obj["sha256"] = cache.digest({k: v for k, v in obj.items() if k != "sha256"})
        self.path.write_text(json.dumps(obj))
        self.assertEqual(cache.run(self.csv, self.path)["cache"], "refreshed")

    def test_short_history_rejected(self):
        lines = self.csv.read_text().splitlines()
        self.csv.write_text("\n".join(lines[:20]) + "\n")
        with self.assertRaisesRegex(ValueError, "twenty"):
            cache.run(self.csv, self.path)


if __name__ == "__main__":
    unittest.main()
