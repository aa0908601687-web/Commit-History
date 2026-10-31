import csv,tempfile,unittest
from pathlib import Path
from v6_stacking_e2e import run
class StackingE2ETest(unittest.TestCase):
    def test_csv_training_qaoa_and_sha(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"synthetic.csv"
            with p.open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(["draw_id","first","second"])
                for i in range(16):
                    vals=[(i+6*j)%38+1 for j in range(6)]
                    w.writerow([str(i)," ".join(map(str,vals)),i%8+1])
            result=run(p)
            self.assertEqual(result["data_rows"],16)
            self.assertEqual(len(result["prediction"]["first"]),6)
            self.assertEqual(len(result["sha256"]),64)
            self.assertTrue(result["prediction"]["meta_training_only_on_past_draws"])
            self.assertFalse(result["real_historical_accuracy_verified"])
            self.assertEqual(result,run(p))
if __name__=="__main__":unittest.main()
