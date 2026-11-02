"""Offline coverage for isolated AutoML data preparation; no TensorFlow needed."""
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import superintelligence_automl_research as automl


class AutoMLDataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "draws.csv"
        self.draws = [
            {"draw_id": f"115000{i:03d}", "first": [1, 2, 3, 4, 5, 6],
             "second": 1 + i % 8}
            for i in range(1, 13)
        ]

    def write(self, draws):
        with self.path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(("draw_id", "first", "second"))
            for draw in draws:
                writer.writerow((draw["draw_id"], " ".join(map(str, draw["first"])),
                                 draw["second"]))

    def test_load_and_dataset_are_leakage_aware(self):
        self.write(self.draws)
        loaded = automl.load_draws(self.path)
        self.assertEqual(loaded, self.draws)
        x, first_targets, second_targets = automl.dataset(loaded)
        self.assertEqual((len(x), len(first_targets), len(second_targets)), (2, 2, 2))
        self.assertEqual(len(x[0]), 92)
        self.assertEqual(x[0], automl.features(loaded[:10]))
        self.assertEqual(first_targets[0][0:6], [1.0] * 6)
        self.assertEqual(sum(first_targets[0]), 6)
        self.assertEqual(second_targets[0], loaded[10]["second"] - 1)

    def test_frequency_features_use_only_latest_5_and_10(self):
        self.assertEqual(len(automl.features(self.draws)), 92)
        changed = [dict(row) for row in self.draws]
        changed[0] = {"draw_id": "115000001", "first": [7, 8, 9, 10, 11, 12],
                      "second": 8}
        self.assertEqual(automl.features(changed), automl.features(self.draws))
        f = automl.features(self.draws)
        self.assertAlmostEqual(f[0], 1.0)
        self.assertAlmostEqual(f[38], sum(x["second"] == 1 for x in self.draws[-5:]) / 5)

    def test_main_emits_json_with_mocked_training(self):
        """The CLI contract is tested without importing TensorFlow."""
        result = {"mode": "offline_mock", "prediction_integration": False}
        output = Path(self.temp.name) / "output.json"
        with patch.object(automl, "load_draws", return_value=self.draws) as loader, \
             patch.object(automl, "run", return_value=result) as trainer, \
             patch.object(sys, "argv", ["automl", "--csv", str(self.path),
                                        "--output", str(output), "--workdir", self.temp.name]), \
             redirect_stdout(io.StringIO()) as printed:
            automl.main()
        loader.assert_called_once_with(str(self.path))
        trainer.assert_called_once_with(self.draws, Path(self.temp.name))
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
        self.assertEqual(json.loads(printed.getvalue()), result)

    def test_run_isolated_training_contract_with_fake_numpy_tensorflow_tuner(self):
        """Exercise training orchestration using fake modules; no real TensorFlow."""
        import types
        import numpy as np
        import random
        class FakeTensor:
            def __init__(self, value=None):
                self.value = value
            def __call__(self, value):
                return FakeTensor(value)
        class FakeHP:
            def Choice(self, name, values):
                self.choice = (name, values)
                return values[0]
        class FakeModel:
            def compile(self, **kwargs):
                self.compiled = kwargs
            def predict(self, values, verbose=0):
                return ([[float(i) / 38 for i in range(38)]],
                        [[float(i) / 8 for i in range(8)]])
        model = FakeModel()
        keras = types.SimpleNamespace(
            Input=lambda **kwargs: FakeTensor(),
            layers=types.SimpleNamespace(Dense=lambda *args, **kwargs: FakeTensor()),
            Model=lambda *args: model)
        tf = types.SimpleNamespace(
            keras=keras, random=types.SimpleNamespace(set_seed=lambda seed: None))
        class FakeTuner:
            def __init__(self, build, **kwargs):
                self.model = build(FakeHP())
            def search(self, *args, **kwargs):
                self.searched = True
            def get_best_models(self, count):
                return [self.model]
        kt = types.SimpleNamespace(RandomSearch=FakeTuner)
        import sys
        with patch.dict(sys.modules, {"tensorflow": tf, "keras_tuner": kt}), \
             patch.object(automl.random, "seed") as seed:
            result = automl.run(self.draws, Path(self.temp.name))
        seed.assert_called_once_with(automl.SEED)
        self.assertTrue(result["tensorflow_automl_executed"])
        self.assertFalse(result["prediction_integration"])
        self.assertEqual(result["training_samples"], 2)
        self.assertEqual(result["first"], [33, 34, 35, 36, 37, 38])
        self.assertEqual(result["second"], 8)
        self.assertEqual(len(result["sha256"]), 64)

    def test_short_history_rejected(self):
        self.write(self.draws[:11])
        with self.assertRaisesRegex(ValueError, "At least 12"):
            automl.load_draws(self.path)

    def test_duplicate_and_out_of_range_first_zone_rejected(self):
        for first in ([1, 1, 3, 4, 5, 6], [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5, 39]):
            bad = [dict(x) for x in self.draws]
            bad[-1] = dict(bad[-1], first=first)
            self.write(bad)
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, "Invalid first"):
                automl.load_draws(self.path)

    def test_second_zone_bounds_rejected(self):
        for second in (0, 9):
            bad = [dict(x) for x in self.draws]
            bad[-1] = dict(bad[-1], second=second)
            self.write(bad)
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "Invalid second"):
                automl.load_draws(self.path)


if __name__ == "__main__":
    unittest.main()
