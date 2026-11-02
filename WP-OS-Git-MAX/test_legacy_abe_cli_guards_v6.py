"""Isolated CLI and configuration guard tests for retired ABE modules.

No production V6 integration, pretrained downloads, or real neural training.
"""
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import abe_infer_foundation_models as foundation
import abe_merge_real_models as merger
import abe_partial_neural_bridge as partial
import abe_train_lstm_gru as trainer
import abe_ultra_hybrid_max as hybrid


class RetiredEntrypointBoundaries(unittest.TestCase):
    def test_all_four_main_guards_reject_missing_arguments(self):
        for module in (foundation, merger, partial, trainer):
            with self.subTest(module=module.__name__):
                with patch.object(sys, "argv", [str(module.__file__)]):
                    with self.assertRaises(SystemExit) as caught:
                        runpy.run_path(str(module.__file__), run_name="__main__")
                    self.assertEqual(caught.exception.code, 2)

    def test_hybrid_main_guard_rejects_missing_csv(self):
        with patch.object(sys, "argv", [str(hybrid.__file__)]):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(hybrid.__file__), run_name="__main__")
            self.assertEqual(caught.exception.code, 2)

    def test_hybrid_rejects_unavailable_core_without_replacing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({
                "data_window_draws": 20,
                "components": {"abe_bayesian": "disabled", "lstm": "not_executed"}
            }), encoding="utf-8")
            draws = [{"draw_id": str(i), "first": [1,2,3,4,5,6], "second": 1}
                     for i in range(20)]
            with patch.object(hybrid, "CONFIG", config):
                with self.assertRaisesRegex(RuntimeError, "core unavailable"):
                    hybrid.run(draws)


if __name__ == "__main__":
    unittest.main()
