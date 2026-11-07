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



def mask_to_selection(pool, mask):
    """Bit i corresponds exactly to candidate pool[i]."""
    validate_pool(pool)
    if type(mask) is not int or not 0 <= mask < 256 or mask.bit_count() != 6:
        raise ValueError("expected an 8-bit mask with exactly six selected bits")
    return tuple(pool[i] for i in range(8) if (mask >> i) & 1)


def exact_mask_argmax(pool, score):
    """Exhaustively optimize the same score function over all 28 valid masks."""
    validate_pool(pool)
    masks = [m for m in range(256) if m.bit_count() == 6]
    return max(masks, key=lambda m: (score(mask_to_selection(pool, m)), -m))


class BitMappingAndObjectiveTests(unittest.TestCase):
    def test_all_masks_bijection(self):
        pool = (1, 3, 8, 12, 19, 24, 31, 38)
        masks = [m for m in range(256) if m.bit_count() == 6]
        self.assertEqual(len(masks), 28)
        self.assertEqual(set(map(lambda m: mask_to_selection(pool, m), masks)),
                         set(enumerate_six(pool)))

    def test_mask_objective_matches_exact_search(self):
        pool = (1, 3, 8, 12, 19, 24, 31, 38)
        score = lambda xs: sum(x * x for x in xs) + (10 if 1 in xs and 38 in xs else 0)
        winning_mask = exact_mask_argmax(pool, score)
        self.assertEqual(score(mask_to_selection(pool, winning_mask)),
                         score(exact_argmax(pool, score)))

    def test_invalid_masks(self):
        pool = (1, 3, 8, 12, 19, 24, 31, 38)
        for mask in [-1, 256, 0, 255, True, 1.5]:
            with self.assertRaises(ValueError):
                mask_to_selection(pool, mask)

if __name__ == "__main__":
    unittest.main()
