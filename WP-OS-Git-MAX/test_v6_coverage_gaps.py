"""Focused regression tests for V6 OpenFE bridge and three-layer cache.

Dependency-heavy feature training is mocked; integration remains covered separately
by the existing OpenFE and cache E2E workflows.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v6_of3q_cache_v3 as cache
import v6_openfe_bridge as bridge


def sample_rows(n=24):
    return [(f"115000{i:03d}", list(range(1 + i % 30, 7 + i % 30)), (i % 8) + 1) for i in range(1, n + 1)]


class CacheCoverageTests(unittest.TestCase):
    def test_digest_and_tampering_and_key_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.json"
            self.assertIsNone(cache._read(p, {"layer": "data"}))
            written = cache._write(p, {"layer": "data", "items": {1: [2, 3]}})
            self.assertEqual(written["sha256"], cache.digest({"layer": "data", "items": {1: [2, 3]}}))
            self.assertEqual(cache._read(p, {"layer": "data"})["items"], {"1": [2, 3]})
            self.assertIsNone(cache._read(p, {"layer": "other"}))
            p.write_text("{bad json", encoding="utf-8")
            self.assertIsNone(cache._read(p, {"layer": "data"}))
            p.write_text(json.dumps({"layer": "data", "sha256": "wrong"}), encoding="utf-8")
            self.assertIsNone(cache._read(p, {"layer": "data"}))

    def test_layers_refresh_then_hit_and_invalidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv = root / "draws.csv"
            csv.write_text("fixture", encoding="utf-8")
            rows = sample_rows()
            with patch.object(cache, "validate", return_value=rows) as validate, \
                 patch.object(cache, "score_three_history_views", return_value=({1: 0.8, 2: 0.2}, {"engine": "fixture"})) as score, \
                 patch.object(cache, "complete_result", return_value={"first_zone": [1, 2, 3, 4, 5, 6]}) as complete:
                first = cache.run(csv, root / "saved")
                second = cache.run(csv, root / "saved")
                self.assertEqual(first["cache"], {"data": "refreshed", "features": "refreshed", "result": "refreshed"})
                self.assertEqual(second["cache"], {"data": "hit", "features": "hit", "result": "hit"})
                self.assertEqual((validate.call_count, score.call_count, complete.call_count), (1, 1, 1))
                csv.write_text("changed fixture", encoding="utf-8")
                third = cache.run(csv, root / "saved")
                self.assertEqual(third["cache"]["data"], "refreshed")
                self.assertEqual((validate.call_count, score.call_count, complete.call_count), (2, 2, 2))
                (root / "saved" / "features.json").write_text("corrupt", encoding="utf-8")
                fourth = cache.run(csv, root / "saved")
                self.assertEqual(fourth["cache"]["features"], "refreshed")
                self.assertEqual(fourth["cache"]["result"], "hit")

    def test_result_version_and_feature_version_invalidate(self):
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "draws.csv"
            csv.write_text("fixture", encoding="utf-8")
            with patch.object(cache, "validate", return_value=sample_rows()), \
                 patch.object(cache, "score_three_history_views", return_value=({1: .3}, {})), \
                 patch.object(cache, "complete_result", return_value={"ok": True}):
                cache.run(csv, Path(td) / "saved")
                with patch.object(cache, "RESULT_VERSION", "new"):
                    result = cache.run(csv, Path(td) / "saved")
                    self.assertEqual(result["cache"], {"data": "hit", "features": "hit", "result": "refreshed"})
                with patch.object(cache, "FEATURE_VERSION", "new"):
                    result = cache.run(csv, Path(td) / "saved")
                    self.assertEqual(result["cache"]["features"], "refreshed")


class OpenFECoverageTests(unittest.TestCase):
    def test_base_rows_and_view_limits(self):
        rows = sample_rows()
        for limit in (None, 10, 20):
            v = bridge._base_row(rows, 1, limit)
            self.assertEqual(v["number"], 1.0)
            self.assertEqual(v["view_rows"], float(len(rows) if limit is None else limit))
            self.assertIn("freq_10", v)
            self.assertIn("gap_view", v)

    def test_invalid_history(self):
        with self.assertRaises(ValueError):
            bridge.score_first_zone(sample_rows(11))
        with self.assertRaises(ValueError):
            bridge.score_three_history_views(sample_rows(19))

    def test_training_rows_and_no_future_target(self):
        rows = sample_rows(25)
        x, y = bridge._training_frame(rows, three_views=True)
        self.assertEqual(len(x), 5 * 3 * 38)
        self.assertEqual(len(y), len(x))
        x2, y2 = bridge._training_frame(rows)
        self.assertEqual(len(x2), 15 * 38)
        self.assertEqual(len(y2), len(x2))
        with self.assertRaises(ValueError):
            bridge._training_frame([(f"115000{i:03d}", [1, 2, 3, 4, 5, 6], 1) for i in range(1, 12)])

    def test_scoring_three_views_and_legacy(self):
        rows = sample_rows()
        with patch.object(bridge, "_fit_and_score", side_effect=[
            ([.2] * 38, 12, 8),
            ([.1] * 38 + [.2] * 38 + [.3] * 38, 15, 9),
        ]):
            legacy, meta1 = bridge.score_first_zone(rows)
            self.assertEqual(len(legacy), 38)
            self.assertEqual(meta1["used_features"], 8)
            three, meta2 = bridge.score_three_history_views(rows)
            self.assertEqual(len(three), 38)
            self.assertAlmostEqual(three[1], .2)
            self.assertEqual(meta2["view_weights"], {"latest_10": 1 / 3, "latest_20": 1 / 3, "full_archive": 1 / 3})


if __name__ == "__main__":
    unittest.main()
