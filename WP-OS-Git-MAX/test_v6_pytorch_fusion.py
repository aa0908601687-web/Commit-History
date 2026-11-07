"""V6 PyTorch fusion contracts and executable CPU model tests."""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v6_ai_qaoa_e2e as v6

def synthetic_csv(path, count=14):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["draw_id", "first", "second"])
        for i in range(count):
            first = sorted((i * 3 + j * 5) % 38 + 1 for j in range(6))
            writer.writerow([str(115000001 + i), " ".join(f"{n:02d}" for n in first), (i % 8) + 1])

class TorchFusionContracts(unittest.TestCase):
    def test_flag_required_and_invalid_weight(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            v6.run("not-read.csv", pytorch_three_views=True)
        with self.assertRaisesRegex(ValueError, "pytorch_weight"):
            v6.run("not-read.csv", pytorch_weight=1.2)

    def test_fused_scores_reach_actual_candidate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            synthetic_csv(path)
            lightgbm = {n: 0.2 for n in range(1, 39)}
            torch_scores = {n: (0.95 if n >= 31 else 0.05) for n in range(1, 39)}
            expected = list(range(31, 39))
            with patch("v6_openfe_bridge.score_three_history_views", return_value=(lightgbm, {"engine":"OpenFE+LightGBM"})), patch("v6_pytorch_bridge.score_three_history_views_pytorch", return_value=(torch_scores, {"trained":True})):
                result = v6.run(path, openfe_three_views=True, pytorch_three_views=True)
            self.assertEqual(result["candidate_pool"], expected)
            self.assertEqual(result["first_zone"], expected[:6])
            self.assertEqual(result["openfe_feature_layer"]["fusion_weights"], {"lightgbm":0.5, "pytorch":0.5})
            self.assertEqual(result["openfe_feature_layer"]["fused_scores"][31], 0.575)
            self.assertEqual(len(result["first_zone"]), 6)
            self.assertTrue(1 <= result["second_zone"] <= 8)

    def test_invalid_pytorch_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            synthetic_csv(path)
            with patch("v6_openfe_bridge.score_three_history_views", return_value=({n:0.2 for n in range(1,39)}, {})), patch("v6_pytorch_bridge.score_three_history_views_pytorch", return_value=({n: float("nan") for n in range(1,39)}, {})):
                with self.assertRaisesRegex(ValueError, "invalid PyTorch"):
                    v6.run(path, openfe_three_views=True, pytorch_three_views=True)

    def test_real_cpu_training_deterministic_and_complete(self):
        try:
            import torch
            import pandas
            import numpy
        except ImportError:
            self.skipTest("PyTorch/pandas/numpy not installed; CI must install them")
        from v6_pytorch_bridge import score_three_history_views_pytorch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            synthetic_csv(path)
            rows = v6.validate(path)
            a, meta = score_three_history_views_pytorch(rows, epochs=3)
            b, _ = score_three_history_views_pytorch(rows, epochs=3)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(range(1,39)))
        self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in a.values()))
        self.assertEqual(meta["training_rows"], (len(rows)-10)*38*3)
        self.assertEqual(meta["selection_feature_windows"], [5,10])
        self.assertFalse(meta["predictive_advantage_verified"])

if __name__ == "__main__":
    unittest.main()
