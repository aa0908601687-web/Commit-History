"""Offline coverage for retired PyTorch trainer and two legacy CLIs.

No model downloads, real training, ABE selection, or V6 integration.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import abe_train_lstm_gru as trainer
import abe_infer_foundation_models as foundation
import abe_merge_real_models as merge_module
import abe_partial_neural_bridge as bridge_module


def draws(n=25):
    return [{"draw_id": str(i), "first": [1, 2, 3, 4, 5, 6], "second": 1}
            for i in range(n)]


class LegacyRemainingBoundaryTests(unittest.TestCase):
    def test_trainer_rejects_missing_torch_without_installing_it(self):
        real_import = __import__
        def restricted_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch unavailable")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=restricted_import):
            with self.assertRaisesRegex(RuntimeError, "PyTorch required"):
                trainer.run(draws())

    def test_trainer_rejects_short_data_and_invalid_hyperparameters(self):
        fake = types.ModuleType("torch")
        fake.nn = types.SimpleNamespace()
        with patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaisesRegex(ValueError, "At least 25"):
                trainer.run(draws(24))
            for options in ({"epochs": 0}, {"epochs": 201}, {"hidden": 0}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Invalid training"):
                    trainer.run(draws(), **options)

    def test_trainer_cli_does_not_claim_model_execution(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "result.json"
            result = {"through_draw": "24", "mode": "trained_local_lstm_gru_only"}
            with patch.object(sys, "argv", ["trainer", "--csv", "unused",
                                             "--output", str(output), "--epochs", "2", "--seed", "13"]), \
                 patch.object(trainer, "load", return_value=draws()) as loader, \
                 patch.object(trainer, "run", return_value=result) as run:
                trainer.main()
            loader.assert_called_once_with("unused")
            self.assertEqual(run.call_args.kwargs, {"epochs": 2, "seed": 13})
            self.assertEqual(json.loads(output.read_text()), result)

    def test_legacy_cli_guards_are_executable(self):
        for module in (foundation, merge_module, bridge_module):
            with self.subTest(module=module.__name__):
                self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
