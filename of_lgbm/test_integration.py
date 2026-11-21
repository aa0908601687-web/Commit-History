import unittest
from integration import select_power_lotto_from_scores, select_from_model_probabilities


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.first = {n: float(n) / 38 for n in range(1, 39)}
        self.second = {n: float(n) / 8 for n in range(1, 9)}

    def test_full_selection_and_determinism(self):
        result = select_power_lotto_from_scores(self.first, self.second)
        self.assertEqual(result["candidate_pool"], tuple(range(31, 39)))
        self.assertEqual(result["first_area"], tuple(range(33, 39)))
        self.assertEqual(result["second_area"], 8)
        self.assertEqual(result, select_power_lotto_from_scores(self.first, self.second))

    def test_probability_bridge(self):
        result = select_from_model_probabilities(
            list(self.first), list(self.first.values()),
            list(self.second), list(self.second.values()),
        )
        self.assertEqual(result["first_area"], tuple(range(33, 39)))
        self.assertEqual(result["second_area"], 8)

    def test_invalid_number_sets(self):
        with self.assertRaises(ValueError):
            select_power_lotto_from_scores({n: 1 for n in range(1, 38)}, self.second)
        with self.assertRaises(ValueError):
            select_power_lotto_from_scores(self.first, {1: 0.5})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            select_power_lotto_from_scores({**self.first, 38: float("inf")}, self.second)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            select_from_model_probabilities([1, 1], [0.1, 0.2], [1], [0.5])


if __name__ == "__main__":
    unittest.main()
