import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from abe_infer_foundation_models import contexts, score_map


class FoundationContractTests(unittest.TestCase):
    def test_contexts_use_exactly_latest_20_draws(self):
        draws = [{"first": (1,2,3,4,5,6), "second": 2} for _ in range(21)]
        draws[-1] = {"first": (7,8,9,10,11,12), "second": 7}
        values = contexts(draws)
        self.assertEqual(len(values), 46)
        self.assertTrue(all(len(x) == 20 for x in values))
        self.assertEqual(values[0][-1], 0)
        self.assertEqual(values[6][-1], 1)
        self.assertEqual(values[38 + 6][-1], 1)

    def test_scores_cover_both_zones(self):
        output = score_map([0.0] * 46)
        self.assertEqual(len(output["first"]), 38)
        self.assertEqual(len(output["second"]), 8)
        self.assertEqual(output["second"]["8"], 0.5)

    def test_reject_incomplete_nonfinite(self):
        for data in ([0.0] * 45, [float("nan")] * 46, [float("inf")] * 46):
            with self.assertRaises(ValueError):
                score_map(data)

if __name__ == "__main__":
    unittest.main()
