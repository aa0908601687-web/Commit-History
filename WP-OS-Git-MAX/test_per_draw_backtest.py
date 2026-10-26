import unittest

from wp_os_git_max import backtest as main_backtest, predict as main_predict
from wp_os_git_max_10_20 import backtest as control_backtest, predict as control_predict
from wp_os_git_max_fusion import backtest as fusion_backtest, predict as fusion_predict


class PerDrawBacktestTests(unittest.TestCase):
    def setUp(self):
        self.draws = [
            {
                "draw_id": str(115000001 + i),
                "first": tuple(sorted(((i + j * 6) % 38) + 1 for j in range(6))),
                "second": i % 8 + 1,
            }
            for i in range(24)
        ]

    def assert_walk_forward(self, result, start, predictor):
        rows = result["per_draw"]
        self.assertEqual(len(rows), len(self.draws) - start)
        for offset, row in enumerate(rows, start):
            expected = predictor(self.draws[:offset])
            self.assertEqual(row["target_draw"], self.draws[offset]["draw_id"])
            self.assertEqual(row["through_draw"], self.draws[offset - 1]["draw_id"])
            self.assertEqual(row["predicted_first"], expected["first"])
            self.assertEqual(row["predicted_second"], expected["second"])
            self.assertEqual(row["actual_first"], list(self.draws[offset]["first"]))
            self.assertEqual(row["actual_second"], self.draws[offset]["second"])
            self.assertEqual(
                row["first_zone_hits"],
                len(set(expected["first"]) & set(self.draws[offset]["first"])),
            )
            self.assertEqual(
                row["second_zone_hit"],
                int(expected["second"] == self.draws[offset]["second"]),
            )
            self.assertEqual(
                row["jackpot_hit"],
                int(row["first_zone_hits"] == 6 and row["second_zone_hit"] == 1),
            )

    def test_main_has_one_prediction_for_each_target(self):
        self.assert_walk_forward(main_backtest(self.draws), 10, main_predict)

    def test_control_has_one_prediction_for_each_target(self):
        self.assert_walk_forward(control_backtest(self.draws), 20, control_predict)

    def test_fusion_has_one_prediction_for_each_target(self):
        self.assert_walk_forward(fusion_backtest(self.draws), 20, fusion_predict)


if __name__ == "__main__":
    unittest.main()
