import unittest

from wp_os_git_max_10_20 import (
    backtest, compare, predict, rank_draws, rank_second
)


class ControlModelTest(unittest.TestCase):
    def setUp(self):
        self.draws = []
        for index in range(21):
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
        with self.assertRaises(ValueError):
            rank_draws([d["first"] for d in self.draws[:19]], 38)
        with self.assertRaises(ValueError):
            rank_second([d["second"] for d in self.draws[:19]])

    def test_comparison_reports_overlap_and_difference(self):
        control = predict(self.draws)
        main = {"first": control["first"], "second": control["second"]}
        result = compare(main, control)
        self.assertEqual(result["first_overlap_count"], 6)
        self.assertTrue(result["second_same"])
        other = {"first": [33, 34, 35, 36, 37, 38],
                 "second": control["second"] % 8 + 1}
        different = compare(other, control)
        self.assertFalse(different["second_same"])

    def test_walk_forward_backtest_and_empty_rate(self):
        result = backtest(self.draws)
        self.assertEqual(result["tested"], 1)
        self.assertEqual(sum(result["first_zone_hits"].values()), 1)
        self.assertEqual(backtest(self.draws[:20])["tested"], 0)
        self.assertEqual(backtest(self.draws[:20])["first_zone_3plus_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
