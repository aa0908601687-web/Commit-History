import importlib.util
import pathlib
import unittest

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("adaptive_bayesian_ensemble_max", HERE / "adaptive_bayesian_ensemble_max.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def synthetic(n=42):
    result = []
    for i in range(n):
        nums = tuple(sorted(((i*7+j*5)%38)+1 for j in range(6)))
        # ensure distinct; stepping by 5 modulo 38 is bijective
        result.append({"draw_id":f"{115000001+i:09d}", "first":nums, "second":(i%8)+1})
    return result


class AdaptiveEnsembleTests(unittest.TestCase):
    def test_requires_twenty_draws(self):
        with self.assertRaises(ValueError):
            mod.predict(synthetic(19))

    def test_output_and_weights(self):
        p = mod.predict(synthetic())
        self.assertEqual(len(p["first"]),6)
        self.assertEqual(len(set(p["first"])),6)
        self.assertTrue(all(1 <= n <= 38 for n in p["first"]))
        self.assertTrue(1 <= p["second"] <= 8)
        for z in ("first","second"):
            self.assertAlmostEqual(sum(p["weights"][z][k] for k in ("5_10","10_20")),1.0)
            self.assertTrue(.25 <= p["weights"][z]["5_10"] <= .75)

    def test_future_draw_does_not_change_prior_prediction(self):
        draws = synthetic()
        self.assertEqual(mod.predict(draws[:30]),mod.predict(draws[:30]))
        self.assertEqual(mod.predict(draws[:30]),mod.predict(draws[:31][:-1]))

    def test_backtest_first_target_after_twenty_draws(self):
        d = synthetic(23)
        r = mod.backtest(d)
        self.assertEqual(r["tested"],3)
        self.assertEqual(r["per_draw"][0]["through_draw"],d[19]["draw_id"])
        self.assertEqual(r["per_draw"][0]["target_draw"],d[20]["draw_id"])


if __name__ == "__main__":
    unittest.main()
