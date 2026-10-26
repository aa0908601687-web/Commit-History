import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from pathlib import Path

stub = types.ModuleType("cache_and_hit_validation")
stub.digest = lambda obj: __import__("hashlib").sha256(
    json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
stub.load_draws = lambda path: []
sys.modules.setdefault("cache_and_hit_validation", stub)
for module_name in ("wp_os_git_max", "wp_os_git_max_10_20", "wp_os_git_max_fusion"):
    fake = types.ModuleType(module_name)
    fake.predict = lambda draws, name=module_name: {"kind": "prediction", "model": name, "count": len(draws)}
    fake.backtest = lambda draws, name=module_name: {"kind": "backtest", "model": name, "count": len(draws)}
    sys.modules.setdefault(module_name, fake)
spec = importlib.util.spec_from_file_location("static_cache_prewarm", Path(__file__).with_name("static_cache_prewarm.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StaticPrewarmTests(unittest.TestCase):
    def setUp(self):
        self.draws = [{"draw_id": str(115000001 + i), "first": [1, 2, 3, 4, 5, 6], "second": 1} for i in range(20)]
        self.builders = {"prediction/main_5_10": lambda draws: {"count": len(draws)}}

    def test_every_declared_profile_is_prebuilt_and_cache_only(self):
        cache = m.build(self.draws, self.builders)
        self.assertEqual(cache["profiles"], ["prediction/main_5_10"])
        self.assertEqual(m.get_cache_only(cache, "prediction/main_5_10"), {"count": 20})
        with self.assertRaises(m.CacheMissError):
            m.get_cache_only(cache, "prediction/not-prebuilt")

    def test_default_builders_cover_all_six_profiles(self):
        cache = m.build(self.draws)
        self.assertEqual(len(cache["profiles"]), 6)
        self.assertEqual(set(cache["profiles"]), set(m._builders()))

    def test_atomic_round_trip_and_tamper_detection(self):
        cache = m.build(self.draws, self.builders)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.json"
            m.write_atomic(cache, path)
            self.assertEqual(m.load_verified(path), cache)
            changed = json.loads(path.read_text())
            changed["responses"]["prediction/main_5_10"]["count"] = 999
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                m.load_verified(path)

    def test_version_and_manifest_validation(self):
        cache = m.build(self.draws, self.builders)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.json"
            for field, value, message in (
                ("model_set_version", "old", "version"),
                ("profiles", [], "manifest"),
            ):
                changed = dict(cache)
                changed[field] = value
                changed["sha256"] = m.digest({k: v for k, v in changed.items() if k != "sha256"})
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    m.load_verified(path)

    def test_twenty_draws_are_required(self):
        with self.assertRaises(ValueError):
            m.build(self.draws[:19], self.builders)

    def test_cli_prewarm_and_cache_only_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.json"
            with mock.patch.object(m, "load_draws", return_value=self.draws), \
                 mock.patch("sys.argv", ["static_cache_prewarm.py", "--cache", str(path), "--csv", "draws.csv", "--prewarm"]), \
                 redirect_stdout(StringIO()) as output:
                m.main()
            self.assertEqual(json.loads(output.getvalue())["status"], "prewarmed")
            with mock.patch("sys.argv", ["static_cache_prewarm.py", "--cache", str(path), "--profile", "prediction/main_5_10"]), \
                 redirect_stdout(StringIO()) as output:
                m.main()
            self.assertEqual(json.loads(output.getvalue())["kind"], "prediction")

    def test_cli_requires_csv_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.json"
            m.write_atomic(m.build(self.draws), path)
            with mock.patch("sys.argv", ["static_cache_prewarm.py", "--cache", str(path), "--prewarm"]), \
                 self.assertRaises(SystemExit):
                m.main()
            with mock.patch("sys.argv", ["static_cache_prewarm.py", "--cache", str(path)]), \
                 self.assertRaises(SystemExit):
                m.main()


if __name__ == "__main__":
    unittest.main()
