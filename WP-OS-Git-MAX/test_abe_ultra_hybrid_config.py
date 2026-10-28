"""Configuration integrity tests; do not claim deep models are installed."""
import json
import pathlib
import unittest

CONFIG = pathlib.Path(__file__).with_name("abe_ultra_hybrid_max.json")

class TestABEUltraHybridConfig(unittest.TestCase):
    def test_config(self):
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "research_configuration_only")
        self.assertEqual(data["data_window_draws"], 20)
        self.assertAlmostEqual(sum(data["initial_weights"].values()), 1.0)
        self.assertEqual(set(data["initial_weights"]), set(data["components"]))
        self.assertEqual(data["initial_weights"]["t0_gamma"], 0.10)
        self.assertEqual(data["components"]["t0_gamma"], "not_implemented")
        self.assertFalse(data["standalone_transformer_model"])
        self.assertEqual(data["zones"]["first"], {"choose": 6, "range": [1, 38]})
        self.assertEqual(data["zones"]["second"], {"choose": 1, "range": [1, 8]})

if __name__ == "__main__":
    unittest.main()
