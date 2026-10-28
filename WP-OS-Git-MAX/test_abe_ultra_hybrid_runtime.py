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
        self.assertEqual(result["source_draw_count"], 20)
        self.assertEqual(result["feature_window_draws"], 20)
        self.assertEqual(len(result["full_input_sha256"]), 64)
        self.assertEqual(result["full_input_sha256"], result["feature_window_sha256"])
        self.assertFalse(result["input_hash_timestamp_proof"])

    def test_hash_changes_on_older_training_data(self):
        draws = [{"draw_id": str(i), "first": [1, 2, 3, 4, 5, 6], "second": 1}
                 for i in range(1, 22)]
        first = run(draws)
        changed = [dict(d) for d in draws]
        changed[0]["second"] = 2
        second = run(changed)
        self.assertEqual(first["feature_window_sha256"], second["feature_window_sha256"])
        self.assertNotEqual(first["full_input_sha256"], second["full_input_sha256"])

    def test_short_history_rejected(self):
        with self.assertRaises(ValueError):
            run([])


if __name__ == "__main__":
    unittest.main()
