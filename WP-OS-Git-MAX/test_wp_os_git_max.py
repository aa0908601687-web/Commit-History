import unittest
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("wp_os_git_max",Path(__file__).with_name("wp_os_git_max.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.draws=[{"draw_id":str(i),"first":tuple(sorted(((i+j*5)%38)+1 for j in range(6))),"second":i%8+1} for i in range(15)]
    def test_predictions_in_range(self):
        p=m.predict(self.draws[:10])
        self.assertEqual(len(set(p["first"])),6)
        self.assertTrue(all(1<=n<=38 for n in p["first"]))
        self.assertTrue(1<=p["second"]<=8)
    def test_no_future_data(self):
        self.assertEqual(m.predict(self.draws[:10]),m.predict(self.draws[:10]))
    def test_walk_forward(self):
        r=m.backtest(self.draws)
        self.assertEqual(r["tested"],5)
        self.assertEqual(sum(r["first_zone_hits"].values()),5)
    def test_requires_history(self):
        with self.assertRaises(ValueError): m.predict(self.draws[:9])

if __name__=="__main__":
    unittest.main()
