"""Mock upstream TabPrep to cover scoring, view fusion, and error guards offline."""
import math
import sys
import types
import unittest
from unittest.mock import patch
import v6_tabprep_bridge as bridge

ROWS=[(f"115{i:06d}", [1,2,3,4,5,6], 1) for i in range(1,13)]

class FakeFrame:
    def __init__(self, rows):
        self.rows = list(rows)
        self.columns = list(self.rows[0]) if self.rows and isinstance(self.rows[0],dict) else ["feature"]
        self.shape = (len(self.rows), len(self.columns))

class FakeProbabilities:
    def __init__(self, values): self.values = values
    def __getitem__(self, item):
        assert item == (slice(None), 1)
        return self.values

class FakeGenerator:
    mismatch = False
    def __init__(self, **kwargs):
        assert kwargs["use_oofte"] is False
    def fit_transform(self, x, y):
        return FakeFrame([{"feature": 1}] * len(x))
    def transform(self, x):
        name = "other" if self.mismatch else "feature"
        return FakeFrame([{name: 1}] * len(x))

class FakeModel:
    result = [0.25]*114
    def __init__(self, **kwargs): pass
    def fit(self, x, y): pass
    def predict_proba(self, x): return FakeProbabilities(type(self).result)

class TabPrepCoverageTests(unittest.TestCase):
    def execute(self, values=None, mismatch=False):
        FakeGenerator.mismatch=mismatch
        FakeModel.result=[0.25]*114 if values is None else values
        fake_pandas=types.SimpleNamespace(DataFrame=FakeFrame)
        fake_lgbm=types.SimpleNamespace(LGBMClassifier=FakeModel)
        fake_tabprep=types.SimpleNamespace(TabPrepFeatureGenerator=FakeGenerator)
        training=FakeFrame([{"base":1}]*8)
        with patch.dict(sys.modules,{"pandas":fake_pandas,"lightgbm":fake_lgbm,"tabprep":fake_tabprep}), \
             patch.object(bridge,"_training_frame",return_value=(training,[1]*8)), \
             patch.object(bridge,"_base_row",return_value={"base":1}):
            return bridge.score_three_history_views_tabprep(ROWS)

    def test_success_three_views(self):
        scores, meta=self.execute()
        self.assertEqual(len(scores),38)
        self.assertAlmostEqual(scores[1],0.25)
        self.assertEqual(meta["history_views"],list(bridge.VIEW_LIMITS))
        self.assertEqual(meta["generated_feature_columns"],1)
        self.assertTrue(meta["tabprep_executed"])
        self.assertFalse(meta["predictive_advantage_verified"])

    def test_schema_mismatch(self):
        with self.assertRaisesRegex(ValueError,"schema mismatch"):
            self.execute(mismatch=True)

    def test_wrong_prediction_count(self):
        with self.assertRaisesRegex(ValueError,"wrong number"):
            self.execute(values=[0.1]*113)

    def test_nonfinite_and_invalid_probability(self):
        for values in ([math.nan]+[0.1]*113, [-0.5]+[0.1]*113,
                       [1.5]+[0.1]*113):
            with self.subTest(value=values[0]), self.assertRaisesRegex(ValueError,"invalid probability"):
                self.execute(values=values)


if __name__=="__main__":
    unittest.main()
