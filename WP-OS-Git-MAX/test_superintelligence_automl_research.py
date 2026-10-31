import csv, importlib.util, tempfile, unittest
from pathlib import Path
MODULE=Path(__file__).with_name("superintelligence_automl_research.py")
spec=importlib.util.spec_from_file_location("automl_research",MODULE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class AutoMLResearchTests(unittest.TestCase):
    def _csv(self,path,count=12):
        with path.open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["draw_id","first","second"]); w.writeheader()
            for i in range(count):
                nums=[((i+j)%38)+1 for j in range(6)]
                w.writerow({"draw_id":str(i+1),"first":" ".join(map(str,nums)),"second":(i%8)+1})
    def test_load_features_and_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"draws.csv"; self._csv(p)
            draws=m.load_draws(p); x,yf,ys=m.dataset(draws)
            self.assertEqual(len(x),2); self.assertEqual(len(x[0]),92)
            self.assertEqual(sum(yf[0]),6); self.assertIn(ys[0],range(8))
            self.assertEqual(len(m.features(draws)),92)
    def test_requires_twelve_draws(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"draws.csv"; self._csv(p,11)
            with self.assertRaises(ValueError): m.load_draws(p)
    def test_rejects_invalid_zones(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"draws.csv"; self._csv(p)
            text=p.read_text(); p.write_text(text.replace("1 2 3 4 5 6","1 1 2 3 4 5",1))
            with self.assertRaises(ValueError): m.load_draws(p)
if __name__=="__main__": unittest.main()
