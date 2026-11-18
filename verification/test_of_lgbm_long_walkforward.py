"""Long-horizon tests use stubs; real 100-fold ML is intentionally manual."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from of_lgbm_long_walkforward import make_summary, wilson_interval

class LongWalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.ids=[115000001+i for i in range(150)]
        self.draws=[tuple(range(1,7)) for _ in self.ids]
        self.seconds=[1 for _ in self.ids]
    def fake_runner(self,ids,draws,seconds,game,folds):
        self.assertEqual(folds,100)
        outcomes=[]
        for i in range(len(ids)-folds,len(ids)):
            outcomes.append({"target_id":str(ids[i]),"cutoff_id":str(ids[i-1]),"matches":i%4,"random_matches":i%2})
        return {"outcomes":outcomes,"probability_calibration_audit":{"status":"insufficient_prior_predictions"},
                "prediction_error_audit":{"test_only":True}}
    def test_100_exact_sequence_and_summary(self):
        s=make_summary(self.ids,self.draws,self.seconds,"power",100,self.fake_runner)
        self.assertEqual(s["folds"],100)
        self.assertEqual(sum(s["model_0_to_n_matches"].values()),100)
        self.assertEqual(sum(s["sampled_random_0_to_n_matches"].values()),100)
        self.assertFalse(s["predictive_advantage_verified"])
        self.assertFalse(s["official_sources_row_verified"])
        self.assertEqual(s["full_walk_forward"]["outcomes"][0]["cutoff_id"],str(self.ids[49]))
    def test_rejects_too_few_history(self):
        with self.assertRaisesRegex(ValueError,"25 earlier"):
            make_summary(self.ids[:120],self.draws[:120],self.seconds[:120],"power",100,self.fake_runner)
    def test_rejects_unsupported_horizon(self):
        with self.assertRaisesRegex(ValueError,"100, 200, or 500"):
            make_summary(self.ids,self.draws,self.seconds,"power",5,self.fake_runner)
    def test_rejects_future_data_cutoff(self):
        def leak(ids,draws,seconds,game,folds):
            r=self.fake_runner(ids,draws,seconds,game,folds)
            r["outcomes"][0]["cutoff_id"]=r["outcomes"][0]["target_id"]
            return r
        with self.assertRaisesRegex(ValueError,"future-data leakage"):
            make_summary(self.ids,self.draws,self.seconds,"power",100,leak)
    def test_wilson_limits(self):
        self.assertEqual(wilson_interval(0,100)[0],0.)
        self.assertEqual(wilson_interval(100,100)[1],1.)
        with self.assertRaises(ValueError): wilson_interval(101,100)

if __name__=="__main__":unittest.main()
