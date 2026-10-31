import csv,tempfile,unittest
from pathlib import Path
from v6_e2e import load_history
from v6_walkforward import evaluate
class WalkforwardTests(unittest.TestCase):
    def test_no_future_data_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"draws.csv"
            with path.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["draw_id","first","second"])
                for i in range(14):
                    vals=[((i+j*6)%38)+1 for j in range(6)]
                    w.writerow([str(i)," ".join(map(str,vals)),i%8+1])
            result=evaluate(load_history(path))
            self.assertEqual(result["predictions"],4)
            self.assertEqual(sum(result["hit_distribution_0_to_6"]),4)
            self.assertEqual(result["rows"][0]["training_through"],"9")
            self.assertEqual(result["rows"][0]["draw_id"],"10")
            self.assertEqual(result["random_second_hit_rate"],0.125)
            self.assertFalse(result["predictive_advantage_verified"])
if __name__=="__main__":unittest.main()
