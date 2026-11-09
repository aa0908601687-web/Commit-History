import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location("v6_ai_exact_e2e",ROOT/"v6_ai_exact_e2e.py")
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


    def test_three_view_direct_inference_without_cache(self):
        from types import SimpleNamespace
        captured = {}
        def score(rows, n_jobs=1):
            captured["rows"] = rows
            return ({n: float(n) for n in range(1, 39)},
                    {"v3_cache_features_used": False, "history_views": ["latest_5", "latest_10", "full_archive"]})
        fake_bridge = SimpleNamespace(score_three_history_views=score)
        with patch.dict(sys.modules, {"v6_openfe_bridge": fake_bridge}):
            result = mod.run(self.csv, openfe_three_views=True)
        self.assertEqual(len(captured["rows"]), len(mod.validate(self.csv)))
        self.assertFalse(result["openfe_feature_layer"]["v3_cache_features_used"])
        self.assertEqual(result["openfe_feature_layer"]["weight"], 1.0)

    def test_three_view_distinct_features_and_no_cache(self):
        import inspect
        bridge_spec = importlib.util.spec_from_file_location("v6_openfe_bridge", ROOT / "v6_openfe_bridge.py")
        bridge = importlib.util.module_from_spec(bridge_spec)
        bridge_spec.loader.exec_module(bridge)
        self.assertNotIn("v3_cache", inspect.signature(bridge.score_three_history_views).parameters)
        rows = [(str(115000001 + i), [1, 2, 3, 4, 5, 6] if i < 5 else [7, 8, 9, 10, 11, 12], 1) for i in range(10)]
        ten = bridge._base_row(rows, 1, 10)
        archive = bridge._base_row(rows, 1, None)
        self.assertNotEqual(ten["freq_view"], archive["freq_view"])
        self.assertEqual(ten["freq_view"], 0.5)
        self.assertAlmostEqual(archive["freq_view"], 15 / 55)

    def test_number_and_view_alignment(self):
        from types import SimpleNamespace
        bridge_spec = importlib.util.spec_from_file_location("v6_openfe_bridge", ROOT / "v6_openfe_bridge.py")
        bridge = importlib.util.module_from_spec(bridge_spec)
        bridge_spec.loader.exec_module(bridge)
        rows = mod.validate(self.csv)
        def fake_fit(train_x, train_y, current_x, **kwargs):
            return [float(i) for i in range(114)], 0, 0
        class FakeFrame(list):
            pass
        fake_pd = SimpleNamespace(DataFrame=lambda records: FakeFrame(records))
        with patch.dict(sys.modules, {"pandas": fake_pd}), \
             patch.object(bridge, "_training_frame", return_value=([None] * 114, None)), \
             patch.object(bridge, "_fit_and_score", side_effect=fake_fit):
            scores, meta = bridge.score_three_history_views(rows)
        self.assertEqual(len(scores), 38)
        self.assertAlmostEqual(scores[1], (0 + 38 + 76) / 3)
        self.assertAlmostEqual(scores[38], (37 + 75 + 113) / 3)
        self.assertFalse(meta["v3_cache_features_used"])

if __name__=="__main__":unittest.main()
