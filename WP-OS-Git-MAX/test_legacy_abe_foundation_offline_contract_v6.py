"""Legacy opt-in foundation inference contracts; no network, ML downloads or V6 integration."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import abe_infer_foundation_models as foundation


class LegacyFoundationOfflineContracts(unittest.TestCase):
    def test_contexts_reject_short_archive(self):
        with self.assertRaisesRegex(ValueError, "20 completed"):
            foundation.contexts([{"first": [1], "second": 1}] * 19)

    def test_contexts_keep_last_twenty_and_two_zones(self):
        old = {"first": [1], "second": 1}
        new = {"first": [2, 38], "second": 8}
        matrix = foundation.contexts([old] + [new] * 20)
        self.assertEqual((len(matrix), len(matrix[0])), (46, 20))
        self.assertEqual(matrix[0], [0.0] * 20)
        self.assertEqual(matrix[1], [1.0] * 20)
        self.assertEqual(matrix[37], [1.0] * 20)
        self.assertEqual(matrix[38], [0.0] * 20)
        self.assertEqual(matrix[45], [1.0] * 20)

    def test_score_map_rejects_wrong_size_or_nonfinite(self):
        for values in ([0.0] * 45, [0.0] * 47, [0.0] * 45 + [float("nan")],
                       [0.0] * 45 + [float("inf")]):
            with self.subTest(length=len(values)):
                with self.assertRaises(ValueError):
                    foundation.score_map(values)

    def test_score_map_bounds_and_zone_alignment(self):
        scores = foundation.score_map([-1000.0] + [0.0] * 37 + [1000.0] + [0.0] * 7)
        self.assertEqual(len(scores["first"]), 38)
        self.assertEqual(len(scores["second"]), 8)
        self.assertGreater(scores["first"]["1"], 0.0)
        self.assertLess(scores["first"]["1"], 0.001)
        self.assertGreater(scores["second"]["1"], 0.999)
        self.assertAlmostEqual(scores["second"]["8"], 0.5)

    def test_main_writes_provenance_only_with_mocked_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "foundation.json"
            draws = [{"draw_id": str(i), "first": [1, 2, 3, 4, 5, 6],
                      "second": 1} for i in range(20)]
            table = foundation.score_map([0.0] * 46)
            with patch.object(sys, "argv", ["foundation", "--csv", "unused.csv",
                                             "--output", str(output)]), \
                 patch.object(foundation, "load", return_value=draws), \
                 patch.object(foundation, "infer_chronos", return_value=table) as chronos, \
                 patch.object(foundation, "infer_t0", return_value=table) as t0:
                foundation.main()
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["through_draw"], "19")
            self.assertEqual(set(data["models"]), {"chronos_2", "t0_gamma"})
            self.assertEqual(data["execution_mode"], "real_pretrained_inference")
            chronos.assert_called_once()
            t0.assert_called_once()


if __name__ == "__main__":
    unittest.main()
