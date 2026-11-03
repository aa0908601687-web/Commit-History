"""Focused regression tests for V6 OpenFE bridge and three-layer cache.

Dependency-heavy feature training is mocked; integration remains covered separately
by the existing OpenFE and cache E2E workflows.
"""
import json
import runpy
import sys
import tempfile
from types import SimpleNamespace
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
        for limit in (None, 5, 10):
            v = bridge._base_row(rows, 1, limit)
            self.assertEqual(v["number"], 1.0)
            self.assertEqual(v["view_rows"], float(10 if limit is None else limit))
            self.assertIn("freq_10", v)
            self.assertIn("gap_view", v)

    def test_invalid_history(self):
        with self.assertRaises(ValueError):
            bridge.score_first_zone(sample_rows(11))
        with self.assertRaises(ValueError):
            bridge.score_three_history_views(sample_rows(11))

    def test_training_rows_and_no_future_target(self):
        rows = sample_rows(25)
        fake_pandas = SimpleNamespace(DataFrame=list, Series=lambda values, name=None: values)
        with patch.dict(sys.modules, {"pandas": fake_pandas}):
            x, y = bridge._training_frame(rows, three_views=True)
            self.assertEqual(len(x), 15 * 3 * 38)
            self.assertEqual(len(y), len(x))
            x2, y2 = bridge._training_frame(rows)
            self.assertEqual(len(x2), 15 * 38)
            self.assertEqual(len(y2), len(x2))
            with self.assertRaises(ValueError):
                bridge._training_frame([(f"115000{i:03d}", [], 1) for i in range(1, 12)])

    def test_scoring_three_views_and_legacy(self):
        rows = sample_rows()
        with patch.dict(sys.modules, {"pandas": SimpleNamespace(DataFrame=list, Series=lambda values, name=None: values)}), patch.object(bridge, "_fit_and_score", side_effect=[
            ([.2] * 38, 12, 8),
            ([.1] * 38 + [.2] * 38 + [.3] * 38, 15, 9),
        ]):
            legacy, meta1 = bridge.score_first_zone(rows)
            self.assertEqual(len(legacy), 38)
            self.assertEqual(meta1["used_features"], 8)
            three, meta2 = bridge.score_three_history_views(rows)
            self.assertEqual(len(three), 38)
            self.assertAlmostEqual(three[1], .2)
            self.assertEqual(meta2["view_weights"], {"latest_5": 1 / 3, "latest_10": 1 / 3, "full_archive": 1 / 3})


    def test_anchor_sampling_large_history(self):
        rows = sample_rows(110)
        fake_pandas = SimpleNamespace(DataFrame=list, Series=lambda values, name=None: values)
        with patch.dict(sys.modules, {"pandas": fake_pandas}):
            x, y = bridge._training_frame(rows, three_views=True, max_anchors=8)
        self.assertEqual(len(x), 8 * 3 * 38)
        self.assertEqual(len(y), len(x))

    def test_fit_scoring_with_fake_dependencies(self):
        class FakeRandom:
            @staticmethod
            def seed(value):
                assert value == bridge.SEED

        class FakeOpenFE:
            def fit(self, **kwargs):
                return ["feat1", "feat2"]

        class FakeClassifier:
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y):
                self.x, self.y = x, y
            def predict_proba(self, x):
                class FakeProbabilities:
                    def __getitem__(self, index):
                        if index == (slice(None), 1):
                            return [0.75 for _ in x]
                        raise IndexError(index)
                return FakeProbabilities()

        fake_np = SimpleNamespace(random=FakeRandom())
        fake_openfe = SimpleNamespace(OpenFE=FakeOpenFE, transform=lambda x, y, selected, n_jobs: (x, y))
        fake_lgbm = SimpleNamespace(LGBMClassifier=FakeClassifier)
        with patch.dict(sys.modules, {"numpy": fake_np, "openfe": fake_openfe, "lightgbm": fake_lgbm}):
            for cap, expected in [(1, 1), (0, 0)]:
                scores, generated, used = bridge._fit_and_score(
                    [{"n": 1}], [1], [{"n": 2}], n_jobs=1, max_features=cap
                )
                self.assertEqual((generated, used), (2, expected))
                self.assertEqual(scores[0], 0.75)

    def test_cache_cli_output_and_stdout(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "result.json"
            with patch.object(cache, "run", return_value={"ok": True}), \
                 patch.object(sys, "argv", ["cache", "--csv", "data.csv", "--cache-dir", td, "--output", str(output)]):
                cache.main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})
            with patch.object(cache, "run", return_value={"ok": True}), \
                 patch.object(sys, "argv", ["cache", "--csv", "data.csv", "--cache-dir", td]):
                cache.main()


    def test_training_skips_history_view_longer_than_available(self):
        fake_pandas = SimpleNamespace(DataFrame=list, Series=lambda values, name=None: values)
        with patch.dict(sys.modules, {"pandas": fake_pandas}), \
             patch.object(bridge, "VIEW_LIMITS", {"latest_10": 10, "long_view": 100, "full_archive": None}):
            x, y = bridge._training_frame(sample_rows(25), three_views=True)
        self.assertEqual(len(x), 15 * 2 * 38)
        self.assertEqual(len(y), len(x))

    def test_qaoa_module_entrypoint(self):
        import v6_ai_qaoa_e2e as qaoa
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "draws.csv"
            csv.write_text("draw_id,first,second\n" + "".join(
                f"115000{i:03d},1 2 3 4 5 6,{i % 8 + 1}\n" for i in range(1, 11)
            ), encoding="utf-8")
            with patch.object(sys, "argv", ["v6_ai_qaoa_e2e.py", "--csv", str(csv)]):
                runpy.run_path(qaoa.__file__, run_name="__main__")


    def test_qaoa_csv_reader_exceptional_exit(self):
        import v6_ai_qaoa_e2e as qaoa
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "draws.csv"
            csv.write_text("draw_id,first,second\n" + "".join(
                f"115000{i:03d},1 2 3 4 5 6,1\n" for i in range(1, 11)
            ), encoding="utf-8")
            original = qaoa.csv.DictReader
            def failing_reader(file):
                reader = original(file)
                def iterate():
                    next(reader)
                    raise RuntimeError("synthetic CSV reader exception")
                    yield
                class FailingReader:
                    fieldnames = reader.fieldnames
                    def __iter__(self):
                        return iterate()
                return FailingReader()
            # The context manager must close its file when the CSV reader raises.
            with patch.object(qaoa.csv, "DictReader", side_effect=failing_reader):
                with self.assertRaises(RuntimeError):
                    qaoa.validate(csv)


    def test_qaoa_csv_file_context_enter_failure(self):
        import v6_ai_qaoa_e2e as qaoa
        from unittest.mock import mock_open
        source = mock_open(read_data="draw_id,first,second\n")()
        source.__enter__.side_effect = OSError("synthetic enter failure")
        with patch("builtins.open", return_value=source):
            with self.assertRaisesRegex(OSError, "synthetic enter failure"):
                qaoa.validate("unreadable.csv")

if __name__ == "__main__":
    unittest.main()
