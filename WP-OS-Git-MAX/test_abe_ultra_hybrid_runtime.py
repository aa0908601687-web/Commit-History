"""Smoke and provenance checks for honest hybrid ABE fallback."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from abe_ultra_hybrid_max import run


class TestHybrid(unittest.TestCase):
    def test_fallback_does_not_claim_neural_inference(self):
        draws = [{"draw_id": str(i), "first": [1, 2, 3, 4, 5, 6], "second": 1}
                 for i in range(1, 21)]
        result = run(draws)
        self.assertEqual(result["execution_mode"], "abe_only_fallback")
        self.assertFalse(result["full_five_model_fusion_executed"])
        self.assertEqual(set(result["unavailable_models"]),
                         {"lstm", "gru", "chronos_2", "t0_gamma"})
        self.assertEqual(len(result["first"]), 6)
        self.assertTrue(1 <= result["second"] <= 8)
        self.assertEqual(result["through_draw"], "20")

    def test_short_history_rejected(self):
        with self.assertRaises(ValueError):
            run([])


if __name__ == "__main__":
    unittest.main()
