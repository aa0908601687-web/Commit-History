import unittest
import pandas as pd
from kumo_csv_predict import prepare

class KumoCsvTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame({"time":range(30),"x":[float(x) for x in range(30)],"target":[x%2 for x in range(30)]})
    def test_valid(self):
        self.assertEqual(len(prepare(self.df,"target","time",20)),30)
    def test_time_leakage_guard(self):
        df=self.df.copy()
        df.loc[5,"time"]=100
        with self.assertRaises(ValueError):
            prepare(df,"target","time",20)
    def test_missing_feature(self):
        df=self.df.copy()
        df.loc[0,"x"]=float("nan")
        with self.assertRaises(ValueError):
            prepare(df,"target","time",20)
if __name__=="__main__":
    unittest.main()
