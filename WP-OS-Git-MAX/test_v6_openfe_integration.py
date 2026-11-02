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
        result=mod.run(cls.csv)
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

if __name__=="__main__":unittest.main()
