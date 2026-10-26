import unittest
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("wp_os_git_max",Path(__file__).with_name("wp_os_git_max.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.draws=[{"draw_id":str(115000001+i),"first":tuple(sorted(((i+j*5)%38)+1 for j in range(6))),"second":i%8+1} for i in range(15)]
    def test_predictions_in_range(self):
        p=m.predict(self.draws[:10])
        self.assertEqual(len(set(p["first"])),6)
        self.assertTrue(all(1<=n<=38 for n in p["first"]))
        self.assertTrue(1<=p["second"]<=8)
    def test_no_future_data(self):
        from unittest.mock import patch
        seen = []
        original = m.predict
        def checked(history):
            end = 10 + len(seen)
            self.assertEqual(history, self.draws[:end])
            seen.append(end)
            return original(history)
        with patch.object(m, "predict", side_effect=checked):
            m.backtest(self.draws)
        self.assertEqual(seen, list(range(10, 15)))
    def test_only_last_ten_affect_prediction(self):
        altered = [dict(d) for d in self.draws]
        altered[0]["first"] = (1, 2, 3, 4, 5, 6)
        altered[0]["second"] = 8
        self.assertEqual(m.predict(self.draws), m.predict(altered))
    def test_loader_rejects_reverse_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "bad.csv"
            p.write_text("draw_id,first,second\n115000002,01 02 03 04 05 06,01\n115000001,01 02 03 04 05 06,01\n")
            with self.assertRaises(ValueError):
                m.load(p)
    def test_walk_forward(self):
        r=m.backtest(self.draws)
        self.assertEqual(r["tested"],5)
        self.assertEqual(sum(r["first_zone_hits"].values()),5)
        self.assertAlmostEqual(sum(r["first_zone_hit_rates"].values()),1.0)
        self.assertAlmostEqual(r["first_zone_3plus_rate"],
                               sum(r["first_zone_hits"][str(k)] for k in range(3,7))/5)
        self.assertAlmostEqual(r["second_zone_hit_rate"],r["second_zone_hits"]/5)
        self.assertAlmostEqual(r["jackpot_hit_rate"],r["jackpot_hits"]/5)
    def test_zero_backtest_rates(self):
        r=m.backtest(self.draws[:10])
        self.assertEqual(r["tested"],0)
        self.assertEqual(r["first_zone_3plus_rate"],0.0)
        self.assertEqual(r["second_zone_hit_rate"],0.0)
        self.assertEqual(r["jackpot_hit_rate"],0.0)
    def test_requires_history(self):
        with self.assertRaises(ValueError): m.predict(self.draws[:9])

if __name__=="__main__":
    unittest.main()

