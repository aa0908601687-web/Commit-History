import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location("v6_ai_qaoa_e2e",ROOT/"v6_ai_qaoa_e2e.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class V6OpenFEIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv=ROOT/"historical_archive.csv"

    def test_legacy_path_stays_disabled(self):
        result=mod.run(self.csv)
        self.assertFalse(result["openfe_feature_layer"]["enabled"])

    def test_weight_guard(self):
        with self.assertRaises(ValueError):
            mod.run(self.csv,True,1.1)

    def test_openfe_scores_feed_ranking(self):
        class Fake:
            @staticmethod
            def score_first_zone(rows,n_jobs=1):
                return ({n:float(n) for n in range(1,39)},
                        {"engine":"fake-openfe","generated_features":2,
                         "used_features":2,"seed":1,"leakage_safe":True})
        with patch.dict(sys.modules,{"v6_openfe_bridge":Fake}):
            result=mod.run(self.csv,True,.30)
        layer=result["openfe_feature_layer"]
        self.assertTrue(layer["enabled"])
        self.assertEqual(layer["weight"],.30)
        self.assertTrue(layer["leakage_safe"])
        self.assertEqual(len(result["first_zone"]),6)

    def test_three_history_views_are_openfe_100_percent(self):
        views=["latest_5","latest_10","full_archive"]
        class Fake:
            @staticmethod
            def score_three_history_views(rows,n_jobs=1):
                per={name:{n:float(n) for n in range(1,39)} for name in views}
                return ({n:float(n) for n in range(1,39)},
                        {"engine":"fake-openfe","mode":"three_history_views_100_percent",
                         "history_views":views,"view_weights":{name:1/3 for name in views},
                         "per_view_scores":per,"leakage_safe":True})
        with patch.dict(sys.modules,{"v6_openfe_bridge":Fake}):
            result=mod.run(self.csv,openfe_three_views=True)
        layer=result["openfe_feature_layer"]
        self.assertEqual(layer["weight"],1.0)
        self.assertEqual(layer["history_views"],views)
        self.assertTrue(set(views).issubset(result["views"]))
        self.assertEqual(len(result["first_zone"]),6)


    def test_three_view_cache_passes_verified_payload(self):
        from types import SimpleNamespace
        cache = {"through_draw": "test-cache"}
        captured = {}
        def score(rows, n_jobs=1, v3_cache=None):
            captured["cache"] = v3_cache
            return ({n: float(n) for n in range(1, 39)}, {"v3_cache_features_used": True})
        fake_bridge = SimpleNamespace(score_three_history_views=score)
        fake_loader = SimpleNamespace(load_draws=lambda path: ["loaded"])
        fake_cache = SimpleNamespace(cache_for=lambda rows, path: (cache, {"verified": True}))
        with patch.dict(sys.modules, {
            "v6_openfe_bridge": fake_bridge,
            "cache_and_hit_validation": fake_loader,
            "cache_v3_pipeline": fake_cache,
        }):
            result = mod.run(self.csv, openfe_three_views=True, v3_cache_path="test-cache.json")
        self.assertIs(captured["cache"], cache)
        self.assertTrue(result["openfe_feature_layer"]["v3_cache_features_used"])

    def test_three_view_cache_features_and_cutoff_guard(self):
        from types import SimpleNamespace
        import v6_openfe_bridge as bridge
        rows = [(str(115000001 + i), [1, 2, 3, 4, 5, 6], 1) for i in range(12)]
        cache = {"through_draw": rows[-1][0],
                 "first": {"five": [2] * 38, "ten": [4] * 38, "age": [2] * 38}}
        seen = {}
        def fit(train_x, train_y, current_x, **kwargs):
            seen["records"] = current_x
            return [0.5] * 114, 2, 2
        fake_pd = SimpleNamespace(DataFrame=lambda records: records)
        with patch.dict(sys.modules, {"pandas": fake_pd}), \
             patch.object(bridge, "_training_frame", return_value=([None] * 114, None)), \
             patch.object(bridge, "_fit_and_score", side_effect=fit):
            with self.assertRaisesRegex(ValueError, "cutoff mismatch"):
                bridge.score_three_history_views(rows, v3_cache={**cache, "through_draw": "wrong"})
            scores, metadata = bridge.score_three_history_views(rows, v3_cache=cache)
        self.assertEqual(len(scores), 38)
        self.assertTrue(metadata["v3_cache_features_used"])
        records = seen["records"]
        self.assertEqual(len(records), 114)
        self.assertEqual(records[0]["freq_view"], 2 / 5)
        self.assertEqual(records[38]["freq_view"], 4 / 10)
        self.assertEqual(records[76]["freq_view"], 4 / 10)
        self.assertEqual(records[38]["gap_view"], 3.0)

if __name__=="__main__":unittest.main()
