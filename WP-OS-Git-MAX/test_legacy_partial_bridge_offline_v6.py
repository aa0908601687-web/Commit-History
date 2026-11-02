"""Offline legacy partial-bridge contracts; intentionally excluded from V6 execution."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import abe_partial_neural_bridge as legacy


def draws():
    return [{"draw_id": f"1150000{i:02}", "first": [1,2,3,4,5,6], "second": 1}
            for i in range(25)]


def payload(cutoff):
    scores = {"first": {str(i): 1.0 for i in range(1,39)},
              "second": {str(i): 1.0 for i in range(1,9)}}
    return {"through_draw": cutoff, "models": {"lstm": scores, "gru": scores}}


class LegacyPartialBridgeTests(unittest.TestCase):
    def test_rejects_wrong_cutoff(self):
        with self.assertRaisesRegex(ValueError, "matching cutoff"):
            legacy.bridge(draws(), payload("bad"))

    def test_rejects_extra_network_model(self):
        p = payload(draws()[-1]["draw_id"])
        p["models"]["chronos_2"] = p["models"]["lstm"]
        with self.assertRaisesRegex(ValueError, "exactly trained"):
            legacy.bridge(draws(), p)

    def test_offline_bridge_uses_neutral_reserve_not_model_predictions(self):
        d = draws()
        p = payload(d[-1]["draw_id"])
        baseline = {"weights": {"first": {"5_10": 0.5}, "second": {"5_10": 0.5}}}
        with patch.object(legacy, "predict", return_value=baseline), \
             patch.object(legacy, "rank_zone", side_effect=lambda seq, upper, a, b, scalar: {i: 1.0 for i in range(1, upper+1)}), \
             patch.object(legacy, "normalized", side_effect=lambda score: {i: 1/len(score) for i in score}):
            out = legacy.bridge(d, p)
        self.assertEqual(out["first"], [1,2,3,4,5,6])
        self.assertEqual(out["second"], 1)
        self.assertEqual(out["execution_mode"], "abe_lstm_gru_partial_with_uniform_reserve")
        self.assertFalse(out["full_five_model_fusion_executed"])
        self.assertEqual(out["effective_weights"]["unallocated_uniform_reserve"], 0.2)

    def test_cli_uses_mocked_archive_and_no_network(self):
        d = draws()
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / "input.json"
            dest = Path(temp) / "out.json"
            src.write_text(json.dumps(payload(d[-1]["draw_id"])), encoding="utf-8")
            canned = {"execution_mode": "offline_test_fixture"}
            with patch.object(sys, "argv", ["partial", "--csv", "unused.csv", "--trained-scores", str(src), "--output", str(dest)]), \
                 patch.object(legacy.base, "load", return_value=d), \
                 patch.object(legacy, "bridge", return_value=canned):
                legacy.main()
            self.assertEqual(json.loads(dest.read_text(encoding="utf-8")), canned)


if __name__ == "__main__":
    unittest.main()
