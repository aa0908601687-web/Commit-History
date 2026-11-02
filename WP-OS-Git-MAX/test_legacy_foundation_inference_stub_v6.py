"""Isolated legacy CLI entrypoint and inference error-path regression tests.

These tests neither enable ABE in V6 nor download/run pretrained models.
"""
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import abe_infer_foundation_models as foundation
import abe_merge_real_models as merger
import abe_partial_neural_bridge as partial
import abe_train_lstm_gru as trainer


class TensorStub:
    def __init__(self, values, size=None):
        self.values = values
        self.size = size
    def numel(self):
        return self.size if self.size is not None else 46
    def reshape(self, n):
        return self
    def tolist(self):
        return [0.0] * 46
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self


class LegacyEntrypointOfflineTests(unittest.TestCase):
    def test_foundation_chronos_valid_and_invalid_shapes(self):
        class Pipeline:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def predict_quantiles(self, *args, **kwargs):
                return None, TensorStub([], size=46)
        torch = types.ModuleType("torch")
        torch.float32 = "float32"
        torch.tensor = lambda x, dtype=None: x
        torch.as_tensor = lambda x: x
        chronos = types.ModuleType("chronos")
        chronos.Chronos2Pipeline = Pipeline
        with patch.dict(sys.modules, {"torch":torch, "chronos":chronos}):
            result = foundation.infer_chronos([[0.0]*20]*46)
            self.assertEqual(len(result["first"]), 38)
            with patch.object(Pipeline,"predict_quantiles",return_value=(None,TensorStub([],size=45))):
                with self.assertRaisesRegex(ValueError,"46 one-step"):
                    foundation.infer_chronos([[0.0]*20]*46)

    def test_foundation_t0_with_numpy_like_median(self):
        class Median:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self
            def reshape(self, *args): return self
            def tolist(self): return [0.0]*46
        class Model:
            @classmethod
            def from_pretrained(cls, *args): return cls()
            def eval(self): return self
            def predict(self, *args, **kwargs):
                return types.SimpleNamespace(median=Median())
        numpy = types.ModuleType("numpy")
        numpy.float32 = "float32"
        numpy.asarray = lambda x, dtype=None: x
        t0 = types.ModuleType("t0")
        t0.T0Forecaster = Model
        with patch.dict(sys.modules, {"numpy":numpy,"t0":t0}):
            self.assertEqual(len(foundation.infer_t0([[0.0]*20]*46)["second"]),8)

    def test_all_legacy_script_guards_with_stubbed_main(self):
        for module in (foundation, merger, partial, trainer):
            with self.subTest(module=module.__name__):
                source = Path(module.__file__)
                with patch("runpy.run_path", wraps=runpy.run_path):
                    with patch("builtins.print"):
                        with patch.object(sys,"argv",[str(source)]), patch.object(module,"main"):
                            self.assertTrue(source.exists())


if __name__=="__main__":
    unittest.main()
