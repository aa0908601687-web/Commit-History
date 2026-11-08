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
    def test_all_256_bitstates_match_penalty_and_reward(self):
        scores = [0.2, -0.7, 0.8, 0.1, -0.3, 0.9, 0.4, 0.05]
        pair = {(0, 1): 0.15, (3, 6): -0.25}
        Q, offset = build_qubo(scores, pair)
        results = []
        for bits in product((0, 1), repeat=8):
            chosen = [i for i, x in enumerate(bits) if x]
            reward = sum(scores[i] for i in chosen) + sum(
                val for (i, j), val in pair.items() if bits[i] and bits[j])
            magnitude = sum(abs(x) for x in scores) + sum(abs(v) for v in pair.values())
            expected = -reward + (magnitude + 1.0)*(len(chosen) - 6)**2
            actual = energy(bits, Q, offset)
            self.assertAlmostEqual(actual, expected, places=9)
            results.append((actual, bits))
        self.assertTrue(all(sum(bits) == 6 for e, bits in results
                            if abs(e - min(v for v, _ in results)) < 1e-8))
        optimal = exact_six(scores, pair)
        winner = tuple(int(i in optimal["indices"]) for i in range(8))
        self.assertTrue(check_solver_output(winner, scores, pair)["optimal"])
        self.assertEqual(optimal["combinations_checked"], 28)

    def test_reject_malformed_solver_bits(self):
        scores = [0.1] * 8
        for bits in ([1] * 7, [1] * 9, [1, 1, 1, 1, 1, 0, 0, 2],
                     [1, 1, 1, 1, 1, 0, 0, True]):
            self.assertEqual(check_solver_output(bits, scores),
                             {"valid": False, "optimal": False})

    def test_reject_invalid(self):
        with self.assertRaises(ValueError):
            build_qubo([1]*7)
        with self.assertRaises(ValueError):
            build_qubo([1]*8, penalty=1)

if __name__ == "__main__":
    unittest.main()
