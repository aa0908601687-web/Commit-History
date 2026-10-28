import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from abe_merge_real_models import merge

class MergeTests(unittest.TestCase):
    def fixture(self):
        scores={"first":{str(n):1. for n in range(1,39)},"second":{str(n):1. for n in range(1,9)}}
        return ({"through_draw":"115000086","mode":"trained_local_lstm_gru_only",
                 "models":{"lstm":scores,"gru":scores}},
                {"through_draw":"115000086","execution_mode":"real_pretrained_inference",
                 "models":{"chronos_2":scores,"t0_gamma":scores}})
    def test_merge(self):
        r=merge(*self.fixture())
        self.assertEqual(set(r["models"]),{"lstm","gru","chronos_2","t0_gamma"})
    def test_cutoff_reject(self):
        a,b=self.fixture()
        b["through_draw"]="115000087"
        with self.assertRaises(ValueError): merge(a,b)
    def test_missing_model_reject(self):
        a,b=self.fixture()
        del b["models"]["t0_gamma"]
        with self.assertRaises(ValueError): merge(a,b)
if __name__=="__main__": unittest.main()
