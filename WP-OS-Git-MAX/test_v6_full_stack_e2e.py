"""Fast unit contracts for the opt-in combined CPU E2E; real runtime runs in CI."""
import unittest
from v6_full_stack_e2e import fuse_scores, normalize

class FullStackContractTests(unittest.TestCase):
    def test_normalize_and_fuse(self):
        a = {n: float(n) for n in range(1,39)}
        b = {n: float(39-n) for n in range(1,39)}
        scores = fuse_scores(a,b,.5)
        self.assertEqual(set(scores), set(range(1,39)))
        self.assertTrue(all(abs(v - .5) < 1e-9 for v in scores.values()))
    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            normalize({1: 1})
        with self.assertRaises(ValueError):
            fuse_scores({n: n for n in range(1,39)}, {n: n for n in range(1,39)},1.5)
    def test_constant_feature_is_finite(self):
        scores = normalize({n: 4.2 for n in range(1,39)})
        self.assertEqual(set(scores.values()), {0.0})

if __name__ == "__main__":
    unittest.main()
