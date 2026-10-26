import unittest

from wp_os_git_max_fusion import backtest, predict


class FusionModelTest(unittest.TestCase):
    def setUp(self):
        self.draws = []
        for index in range(25):
            start = index % 33 + 1
            self.draws.append({
                "draw_id": f"115000{index + 1:03d}",
                "first": tuple(range(start, start + 6)),
                "second": index % 8 + 1,
            })

    def test_deterministic_valid_prediction(self):
        result = predict(self.draws)
        self.assertEqual(result, predict(self.draws))
        self.assertEqual(len(result["first"]), 6)
        self.assertEqual(len(set(result["first"])), 6)
        self.assertEqual(result["feature_windows"], [5, 10, 20])
        self.assertEqual(result["fusion_weights"], {
            "main_5_10": 0.5, "control_10_20": 0.5
        })
        self.assertTrue(1 <= result["second"] <= 8)
        self.assertEqual(result["through_draw"], self.draws[-1]["draw_id"])

    def test_requires_twenty_draws(self):
        with self.assertRaises(ValueError):
            predict(self.draws[:19])

    def test_component_predictions_are_preserved(self):
        components = predict(self.draws)["component_predictions"]
        self.assertIn("main_5_10", components)
        self.assertIn("control_10_20", components)
        self.assertEqual(len(components["main_5_10"]["first"]), 6)
        self.assertEqual(len(components["control_10_20"]["first"]), 6)

    def test_walk_forward_backtest_and_empty_rate(self):
        result = backtest(self.draws)
        self.assertEqual(result["tested"], 5)
        self.assertEqual(sum(result["first_zone_hits"].values()), 5)
        empty = backtest(self.draws[:20])
        self.assertEqual(empty["tested"], 0)
        self.assertEqual(empty["first_zone_3plus_rate"], 0.0)
        self.assertEqual(empty["second_zone_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
