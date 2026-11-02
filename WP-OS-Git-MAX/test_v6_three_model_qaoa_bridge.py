import unittest
from pathlib import Path
from unittest.mock import patch

from v6_three_model_qaoa_bridge import run


class V6BridgeContractTests(unittest.TestCase):
    def setUp(self):
        self.archive = Path(__file__).with_name("historical_archive.csv")

    def test_real_archive_and_models_have_same_cutoff(self):
        result = run(self.archive)
        self.assertEqual(set(result["models"]), {"main_5_10", "control_10_20", "fusion_5_10_20"})
        self.assertEqual(len(result["qaoa_research"]["first"]), 6)
        self.assertEqual(set(result["comparison"]), set(result["models"]))
        self.assertFalse(result["full_e2e_verified"])
        self.assertFalse(result["live_official_api_verified"])

    def test_deterministic_output(self):
        self.assertEqual(run(self.archive), run(self.archive))

    def test_rejects_qaoa_cutoff_mismatch(self):
        from v6_three_model_qaoa_bridge import run_qaoa
        def bad(path):
            payload = run_qaoa(path)
            payload["last_draw"] = "000000000"
            return payload
        with patch("v6_three_model_qaoa_bridge.run_qaoa", side_effect=bad):
            with self.assertRaisesRegex(ValueError, "cutoffs disagree"):
                run(self.archive)


if __name__ == "__main__":
    unittest.main()
