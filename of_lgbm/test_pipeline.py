import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import sha256_file, train_openfe_lightgbm


class SmokeTests(unittest.TestCase):
    def test_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "sample.csv"
            file.write_bytes(b"draw,number\n1,2\n")
            self.assertEqual(
                sha256_file(str(file)),
                hashlib.sha256(file.read_bytes()).hexdigest(),
            )

    def test_empty_split_rejected_before_dependency_import(self):
        with self.assertRaises(ValueError):
            train_openfe_lightgbm([], [], [], [])

    def test_mismatched_labels_rejected(self):
        with self.assertRaises(ValueError):
            train_openfe_lightgbm([1, 2], [0], [3], [1])

    def test_openfe_lightgbm_handoff(self):
        """Verify the isolated adapter wires train/validation through both APIs."""
        calls = []

        class FakeOpenFE:
            def fit(self, *, data, label, n_jobs):
                calls.append(("openfe_fit", data, label, n_jobs))
                return ["generated_feature"]

        def fake_transform(train, valid, features, *, n_jobs):
            calls.append(("transform", train, valid, features, n_jobs))
            return ["train_features"], ["validation_features"]

        class FakeLGBMClassifier:
            def __init__(self, **kwargs):
                calls.append(("model_init", kwargs))

            def fit(self, x, y):
                calls.append(("model_fit", x, y))

            def predict_proba(self, x):
                calls.append(("predict", x))
                return [[0.25, 0.75]]

        fake_openfe = types.ModuleType("openfe")
        fake_openfe.OpenFE = FakeOpenFE
        fake_openfe.transform = fake_transform
        fake_lgb = types.ModuleType("lightgbm")
        fake_lgb.LGBMClassifier = FakeLGBMClassifier

        with patch.dict(sys.modules, {"openfe": fake_openfe, "lightgbm": fake_lgb}):
            result = train_openfe_lightgbm(
                ["past_a", "past_b"], [0, 1],
                ["future_a"], [1], seed=17, n_jobs=2,
            )

        self.assertEqual(result["features"], ["generated_feature"])
        self.assertEqual(result["probabilities"], [[0.25, 0.75]])
        self.assertEqual(calls[0], ("openfe_fit", ["past_a", "past_b"], [0, 1], 2))
        self.assertEqual(
            calls[1],
            ("transform", ["past_a", "past_b"], ["future_a"], ["generated_feature"], 2),
        )
        self.assertIn(("model_fit", ["train_features"], [0, 1]), calls)
        self.assertIn(("predict", ["validation_features"]), calls)


if __name__ == "__main__":
    unittest.main()
