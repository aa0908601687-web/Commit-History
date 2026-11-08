import unittest
from itertools import product
from verification.v6_ising_qubo_reference import build_qubo, energy, exact_six, check_solver_output

class TestQubo(unittest.TestCase):
    def test_exact_top_six(self):
        s = list(range(8))
        result = exact_six(s)
        self.assertEqual(result["indices"], (2, 3, 4, 5, 6, 7))
        self.assertEqual(result["combinations_checked"], 28)
    def test_penalty_enforces_six(self):
        s = [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.1, 0.8]
        q, offset = build_qubo(s, {(0, 1): 0.25})
        winners = min(product((0, 1), repeat=8), key=lambda bits: energy(bits,q,offset))
        self.assertEqual(sum(winners), 6)
    def test_reference_comparison(self):
        s = list(range(8))
        self.assertTrue(check_solver_output((0,0,1,1,1,1,1,1), s)["optimal"])
        self.assertFalse(check_solver_output((1,1,1,1,1,1,0,0), s)["optimal"])
    def test_reject_invalid(self):
        with self.assertRaises(ValueError):
            build_qubo([1]*7)
        with self.assertRaises(ValueError):
            build_qubo([1]*8, penalty=1)

if __name__ == "__main__":
    unittest.main()
