"""Offline tests: no TensorFlow dependency and no future-draw leakage."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).with_name("abe_lstm_gru.py")
spec = importlib.util.spec_from_file_location("abe_lstm_gru_research", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def history(n):
    return [{"draw_id": str(i + 1),
             "first": sorted({((i * 3 + j * 5) % 38) + 1 for j in range(6)}),
             "second": (i % 8) + 1}
            for i in range(n)]


class NeuralResearchTests(unittest.TestCase):
    def test_short_history_falls_back_to_original_abe(self):
        data = history(20)
        self.assertEqual(mod.predict(data)["neural_status"], "skipped_insufficient_history")
        self.assertEqual(mod.predict(data)["first"], mod.abe.predict(data)["first"])
        self.assertEqual(mod.predict(data)["second"], mod.abe.predict(data)["second"])

    def test_only_prior_history_in_neural_samples(self):
        data = history(60)
        xs, ys = mod.examples(data)
        self.assertEqual(len(xs), 40)
        self.assertEqual(xs[0][-1], mod.vector(data[19]))
        self.assertEqual(ys[0], mod.vector(data[20]))
        self.assertEqual(xs[-1][-1], mod.vector(data[58]))
        self.assertEqual(ys[-1], mod.vector(data[59]))

    def test_mocked_neural_fusion_outputs_valid_ticket(self):
        data = history(60)
        fake = {"LSTM": [0.4] * 46, "GRU": [0.6] * 46}
        with patch.object(mod, "train_neural", return_value=(fake, 40)):
            prediction = mod.predict(data)
        self.assertEqual(prediction["neural_status"], "trained")
        self.assertEqual(len(set(prediction["first"])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in prediction["first"]))
        self.assertTrue(1 <= prediction["second"] <= 8)

    def test_neural_sample_floor(self):
        with self.assertRaises(ValueError):
            mod.examples(history(59))

    def test_deterministic_fallback(self):
        self.assertEqual(mod.predict(history(20)), mod.predict(history(20)))


if __name__ == "__main__":
    unittest.main()
