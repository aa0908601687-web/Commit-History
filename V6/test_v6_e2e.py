import csv
import tempfile
import unittest
from pathlib import Path
from v6_e2e import load_history,predict

class EndToEndTests(unittest.TestCase):
    def make_csv(self,where):
        path=Path(where)/"history.csv"
        with path.open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["draw_id","first","second"])
            for k in range(12):
                nums=sorted(((k*3+j*5)%38)+1 for j in range(6))
                writer.writerow([str(100+k)," ".join(map(str,nums)),(k%8)+1])
        return path
    def test_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.make_csv(directory)
            draws=load_history(path)
            result=predict(draws)
            self.assertEqual(len(result["first"]),6)
            self.assertEqual(len(set(result["first"])),6)
            self.assertTrue(all(1<=x<=38 for x in result["first"]))
            self.assertTrue(1<=result["second"]<=8)
            self.assertEqual(result["feature_windows"],[5,10])
            self.assertEqual(result,predict(draws))
            self.assertFalse(result["trained_ai_integrated"])
            self.assertFalse(result["predictive_advantage_verified"])
    def test_reject_duplicate_draw_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.make_csv(directory)
            with path.open("a") as f:f.write("100,1 2 3 4 5 6,1\n")
            with self.assertRaises(ValueError):load_history(path)
if __name__=="__main__":unittest.main()
