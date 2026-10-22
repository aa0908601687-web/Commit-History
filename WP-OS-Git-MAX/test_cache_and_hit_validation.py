import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location("cache_and_hit_validation",Path(__file__).with_name("cache_and_hit_validation.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CacheAndHitTests(unittest.TestCase):
    def setUp(self):
        self.draws=[{"draw_id":str(115000001+i),"first":sorted(((i+j*5)%38)+1 for j in range(6)),"second":i%8+1} for i in range(11)]
    def test_feature_cache(self):
        obj=m.cache_features(self.draws[:10])
        self.assertEqual(obj["through_draw"],"115000010")
        self.assertEqual(sum(obj["first_5"].values()),30)
        self.assertEqual(sum(obj["second_10"].values()),10)
    def test_prediction_snapshot_and_verify(self):
        prediction=m.make_prediction_snapshot(self.draws[:10],"115000011")
        result=m.validate(prediction,self.draws)
        self.assertEqual(result["target_draw"],"115000011")
        self.assertGreaterEqual(result["first_zone_hits"],0)
        self.assertLessEqual(result["first_zone_hits"],6)
    def test_detect_tampering(self):
        p=m.make_prediction_snapshot(self.draws[:10],"115000011")
        p["first"]=[1,2,3,4,5,6]
        with self.assertRaises(ValueError): m.validate(p,self.draws)
    def test_no_future_period(self):
        with self.assertRaises(ValueError):m.make_prediction_snapshot(self.draws[:10],"115000010")
    def test_missing_target(self):
        p=m.make_prediction_snapshot(self.draws[:10],"115000012")
        with self.assertRaises(ValueError):m.validate(p,self.draws)
if __name__=="__main__": unittest.main()
