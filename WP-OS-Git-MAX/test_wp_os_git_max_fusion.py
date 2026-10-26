import unittest

from wp_os_git_max_fusion import predict


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


if __name__ == "__main__":
    unittest.main()
