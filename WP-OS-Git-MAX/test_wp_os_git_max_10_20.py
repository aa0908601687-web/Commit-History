import unittest

from wp_os_git_max_10_20 import compare, predict


class ControlModelTest(unittest.TestCase):
    def setUp(self):
        self.draws = []
        for index in range(20):
            start = index % 33 + 1
            self.draws.append({
                "draw_id": f"115000{index + 1:03d}",
                "first": tuple(range(start, start + 6)),
                "second": index % 8 + 1,
            })

    def test_deterministic_valid_prediction(self):
        one = predict(self.draws)
        self.assertEqual(one, predict(self.draws))
        self.assertEqual(len(one["first"]), 6)
        self.assertEqual(len(set(one["first"])), 6)
        self.assertEqual(one["selection_window"], [10, 20])
        self.assertTrue(1 <= one["second"] <= 8)

    def test_requires_twenty_draws(self):
        with self.assertRaises(ValueError):
            predict(self.draws[:19])

    def test_comparison_reports_overlap(self):
        control = predict(self.draws)
        main = {"first": control["first"], "second": control["second"]}
        result = compare(main, control)
        self.assertEqual(result["first_overlap_count"], 6)
        self.assertTrue(result["second_same"])


if __name__ == "__main__":
    unittest.main()
