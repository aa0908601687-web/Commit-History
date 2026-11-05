"""Independent 8-candidate / choose-6 exhaustive reference checks."""
import itertools
import unittest

def validate_pool(pool):
    if len(pool) != 8 or len(set(pool)) != 8:
        raise ValueError("expected eight distinct candidates")
    if any(type(x) is not int or not 1 <= x <= 38 for x in pool):
        raise ValueError("first-zone candidates must be integers 1..38")
    return tuple(pool)

def enumerate_six(pool):
    return list(itertools.combinations(validate_pool(pool), 6))

def exact_argmax(pool, score):
    choices = enumerate_six(pool)
    return max(choices, key=lambda x: (score(x), tuple(-n for n in x)))

def validate_ticket(first, second):
    return (len(first) == 6 and len(set(first)) == 6
            and all(type(x) is int and 1 <= x <= 38 for x in first)
            and type(second) is int and 1 <= second <= 8)

class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.pool = (1, 3, 8, 12, 19, 24, 31, 38)

    def test_all_28_combinations(self):
        choices = enumerate_six(self.pool)
        self.assertEqual(len(choices), 28)
        self.assertEqual(len(set(choices)), 28)
        self.assertTrue(all(validate_ticket(x, 1) for x in choices))

    def test_exact_argmax(self):
        def score(xs):
            return sum(x * x for x in xs)
        winner = exact_argmax(self.pool, score)
        self.assertTrue(all(score(winner) >= score(x) for x in enumerate_six(self.pool)))

    def test_invalid_pool(self):
        for bad in [(1,) * 8, (0, 1, 2, 3, 4, 5, 6, 7),
                    (1, 2, 3, 4, 5, 6, 7, 39)]:
            with self.assertRaises(ValueError):
                enumerate_six(bad)

    def test_invalid_ticket(self):
        self.assertFalse(validate_ticket((1, 2, 3, 4, 5, 5), 1))
        self.assertFalse(validate_ticket((1, 2, 3, 4, 5, 6), 9))

if __name__ == "__main__":
    unittest.main()
