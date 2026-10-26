import json
import tempfile
import unittest
from pathlib import Path
from cache_v3_pipeline import expected, cache_for, predict_from_cache, snapshot, SCHEMA
from cache_and_hit_validation import validate
from wp_os_git_max import predict as original

class CacheV3Tests(unittest.TestCase):
    def setUp(self):
        self.draws = [{"draw_id":str(115000001+i),
                       "first": sorted(((i+j*5)%38)+1 for j in range(6)),
                       "second":i%8+1} for i in range(12)]
    def test_equivalent_selection_and_hash(self):
        cache = expected(self.draws[:10])
        self.assertEqual(cache["schema"], SCHEMA)
        self.assertEqual(predict_from_cache(cache), original(self.draws[:10]))
    def test_hit_and_new_draw_invalidates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"cache.json"
            one, s1 = cache_for(self.draws[:10], path)
            two, s2 = cache_for(self.draws[:10], path)
            three, s3 = cache_for(self.draws[:11], path)
            self.assertEqual((s1,s2,s3), ("refreshed","hit","refreshed"))
            self.assertEqual(one, two)
            self.assertNotEqual(one["source_sha256"],three["source_sha256"])
    def test_tamper_and_schema_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"cache.json"
            obj,_ = cache_for(self.draws[:10],p)
            obj["first"]["five"][0] = 999
            with self.assertRaises(ValueError): predict_from_cache(obj)
            p.write_text(json.dumps(obj),encoding="utf-8")
            restored,status = cache_for(self.draws[:10],p)
            self.assertEqual(status,"refreshed")
            self.assertNotEqual(restored["first"]["five"][0],999)
            obj = dict(restored,schema=2)
            with self.assertRaises(ValueError): predict_from_cache(obj)
    def test_prediction_snapshot_verification(self):
        prior=self.draws[:10]
        payload=snapshot(prior,predict_from_cache(expected(prior)),"115000011")
        self.assertEqual(validate(payload,self.draws)["target_draw"],"115000011")
        payload["first"]=[1,2,3,4,5,6]
        with self.assertRaises(ValueError): validate(payload,self.draws)

if __name__ == "__main__":
    unittest.main()
