import unittest

from model_bridge import train_and_select, _positive_class_probabilities


class BridgeTests(unittest.TestCase):
    def test_model_to_selection_handoff(self):
        calls = []

        def fake_trainer(train_x, train_y, valid_x, valid_y, *, seed, n_jobs):
            calls.append((len(valid_x), seed, n_jobs))
            return {"probabilities": [[1 - x, x] for x in valid_x]}

        first = [n / 38 for n in range(1, 39)]
        second = [n / 8 for n in range(1, 9)]
        result = train_and_select(
            [0, 1], [0, 1], first, [0] * 38,
            [0, 1], [0, 1], second, [0] * 8,
            first_numbers=list(range(1, 39)), second_numbers=list(range(1, 9)),
            trainer=fake_trainer,
        )
        self.assertEqual(result["first_area"], tuple(range(33, 39)))
        self.assertEqual(result["second_area"], 8)
        self.assertEqual(calls, [(38, 42, 1), (8, 42, 1)])

    def test_reject_missing_candidate(self):
        with self.assertRaises(ValueError):
            train_and_select([], [], [], [], [], [], [], [],
                             first_numbers=list(range(1, 38)),
                             second_numbers=list(range(1, 9)))

    def test_reject_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            _positive_class_probabilities([[0.5, float("nan")]])


if __name__ == "__main__":
    unittest.main()
