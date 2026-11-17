import unittest
import numpy as np
import pandas as pd
from full_pipeline import evaluate, validate

class PipelineTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(3)
        return pd.DataFrame({"time": range(90), "x": rng.normal(size=90), "target": [0,1]*45})
    def test_walk_forward(self):
        rows = evaluate(self.sample(), min_train=50, step=20)
        self.assertTrue(rows)
        self.assertTrue(all(r["train_end"] < r["test_start"] for r in rows))
    def test_duplicate_time_rejected(self):
        df = self.sample()
        df.loc[1,"time"] = 0
        with self.assertRaises(ValueError):
            validate(df, "target", "time")
    def test_target_missing_rejected(self):
        with self.assertRaises(ValueError):
            validate(self.sample(), "missing", "time")

if __name__ == "__main__":
    unittest.main()
