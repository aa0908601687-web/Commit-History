"""Additional isolated merge CLI tests for retired ABE compatibility only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import abe_merge_real_models as merger


def sample():
    zone = {"first": {str(n): 1.0 for n in range(1, 39)},
            "second": {str(n): 1.0 for n in range(1, 9)}}
    return (
        {"through_draw": "115000088", "mode": "trained_local_lstm_gru_only",
         "models": {"lstm": zone, "gru": zone}},
        {"through_draw": "115000088", "execution_mode": "real_pretrained_inference",
         "models": {"chronos_2": zone, "t0_gamma": zone}},
    )


class LegacyMergeCLIContracts(unittest.TestCase):
    def test_rejects_wrong_recurrent_model_mode(self):
        recurrent, foundation = sample()
        recurrent["mode"] = "unverified"
        with self.assertRaisesRegex(ValueError, "Expected trained recurrent"):
            merger.merge(recurrent, foundation)

    def test_rejects_wrong_recurrent_model_keys(self):
        recurrent, foundation = sample()
        recurrent["models"].pop("gru")
        with self.assertRaisesRegex(ValueError, "Expected trained recurrent"):
            merger.merge(recurrent, foundation)

    def test_rejects_wrong_foundation_execution_mode(self):
        recurrent, foundation = sample()
        foundation["execution_mode"] = "mocked"
        with self.assertRaisesRegex(ValueError, "Expected real pretrained"):
            merger.merge(recurrent, foundation)

    def test_cli_serializes_validated_merge_without_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            recurrent, foundation = sample()
            r, f, out = (base / name for name in ("recurrent.json", "foundation.json", "merged.json"))
            r.write_text(json.dumps(recurrent), encoding="utf-8")
            f.write_text(json.dumps(foundation), encoding="utf-8")
            with patch.object(sys, "argv", ["merge", "--recurrent", str(r),
                                             "--foundation", str(f), "--output", str(out)]):
                merger.main()
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["through_draw"], "115000088")
            self.assertEqual(len(result["models"]), 4)
            self.assertEqual(result["input_provenance"]["foundation"], "real_pretrained_inference")


if __name__ == "__main__":
    unittest.main()
