"""Contract tests for the independent-model modular pipeline."""
import copy
import unittest
from unittest.mock import patch

from modular_selection_pipeline import run


class ModularPipelineTests(unittest.TestCase):
    def setUp(self):
        self.draws = [
            {"draw_id": f"{115000001 + i:09d}",
             "first": tuple(sorted(((i * 7 + j * 5) % 38) + 1 for j in range(6))),
             "second": (i % 8) + 1}
            for i in range(20)
        ]

    def test_three_predictions_and_same_cutoff(self):
        result = run(self.draws)
        self.assertEqual(set(result["predictions"]), {"main_5_10", "control_10_20", "fusion_5_10_20"})
        self.assertEqual(result["through_draw"], self.draws[-1]["draw_id"])
        for item in result["predictions"].values():
            self.assertEqual(item["through_draw"], result["through_draw"])

    def test_repeatable_and_no_input_mutation(self):
        before = copy.deepcopy(self.draws)
        self.assertEqual(run(self.draws), run(self.draws))
        self.assertEqual(before, self.draws)

    def test_insufficient_draws_fail_closed(self):
        with self.assertRaises(ValueError):
            run(self.draws[:19])

    def test_model_cutoff_mismatch_is_rejected(self):
        from wp_os_git_max_10_20 import predict as original
        def wrong(draws):
            output = original(draws)
            output["through_draw"] = "000000000"
            return output
        with patch("modular_selection_pipeline.control_predict", wrong):
            with self.assertRaisesRegex(ValueError, "Cutoff mismatch"):
                run(self.draws)


    def test_invalid_predictions_rejected(self):
        from wp_os_git_max_10_20 import predict as original
        changes = [
            {"first": [1, 2]},
            {"first": [1, 1, 2, 3, 4, 5]},
            {"first": [1, 2, 3, 4, 5, 39]},
            {"first": ["1", 2, 3, 4, 5, 6]},
            {"second": 0},
            {"second": "1"},
        ]
        for change in changes:
            with self.subTest(change=change):
                def malformed(draws):
                    payload = original(draws)
                    payload.update(change)
                    return payload
                with patch("modular_selection_pipeline.control_predict", malformed):
                    with self.assertRaisesRegex(ValueError, "Invalid prediction"):
                        run(self.draws)


if __name__ == "__main__":
    unittest.main()
