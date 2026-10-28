"""End-to-end ABE pipeline smoke test with explicitly synthetic four-model scores."""
import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from abe_ultra_hybrid_max import run
from abe_score_adapters import NAMES


class EndToEnd(unittest.TestCase):
    def test_csv_to_fallback_and_synthetic_fusion(self):
        from wp_os_git_max import load
        csv_file = HERE / "historical_latest_20.csv"
        self.assertTrue(csv_file.exists(), "Checked-in 20-draw CSV required")
        draws = load(str(csv_file))
        self.assertGreaterEqual(len(draws), 20)
        cutoff = draws[-1]["draw_id"]
        fallback = run(draws)
        self.assertEqual(fallback["through_draw"], cutoff)
        self.assertEqual(fallback["execution_mode"], "abe_only_fallback")
        self.assertFalse(fallback["full_five_model_fusion_executed"])

        # Uniform tables verify wiring ONLY, not neural model inference.
        payload = {
            "through_draw": cutoff,
            "models": {
                name: {
                    "first": {str(n): 1.0 for n in range(1, 39)},
                    "second": {str(n): 1.0 for n in range(1, 9)}
                } for name in NAMES
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "synthetic_scores.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            fused = run(draws, model_scores_path=path)
            self.assertEqual(fused["execution_mode"], "five_model_external_scores")
            self.assertTrue(fused["five_model_scores_fused"])
            self.assertTrue(fused["external_score_tables_validated"])
            self.assertFalse(fused["neural_inference_verified"])
            self.assertEqual(set(fused["unavailable_models"]), set(NAMES))
            self.assertFalse(fused["full_five_model_fusion_executed"])
            self.assertEqual(fused["through_draw"], cutoff)
            self.assertEqual(len(fused["first"]), 6)
            self.assertEqual(len(set(fused["first"])), 6)
            self.assertTrue(all(1 <= n <= 38 for n in fused["first"]))
            self.assertTrue(1 <= fused["second"] <= 8)

            payload["through_draw"] = "000000000"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                run(draws, model_scores_path=path)


if __name__ == "__main__":
    unittest.main()
