import unittest

from exact_search import select_six_from_eight


class ExactSearchTests(unittest.TestCase):
    def test_selects_best_six(self):
        result = select_six_from_eight({n: float(n) for n in range(1, 9)})
        self.assertEqual(result["numbers"], (3, 4, 5, 6, 7, 8))
        self.assertEqual(result["score"], 33.0)

    def test_deterministic_ties(self):
        result = select_six_from_eight({n: 1 for n in range(1, 9)})
        self.assertEqual(result["numbers"], (1, 2, 3, 4, 5, 6))

    def test_invalid_pool(self):
        with self.assertRaises(ValueError):
            select_six_from_eight({n: 1 for n in range(1, 8)})

    def test_invalid_number(self):
        with self.assertRaises(ValueError):
            select_six_from_eight({n: 1 for n in range(1, 8)} | {39: 1})

    def test_nonfinite_score(self):
        with self.assertRaises(ValueError):
            select_six_from_eight({n: 1 for n in range(1, 8)} | {8: float("nan")})


if __name__ == "__main__":
    unittest.main()
