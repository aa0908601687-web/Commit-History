import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

stub = types.ModuleType("cache_and_hit_validation")
stub.digest = lambda obj: __import__("hashlib").sha256(
    json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()
stub.load_draws = lambda path: []
sys.modules.setdefault("cache_and_hit_validation", stub)
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

    def test_twenty_draws_are_required(self):
        with self.assertRaises(ValueError):
            m.build(self.draws[:19], self.builders)


if __name__ == "__main__":
    unittest.main()
