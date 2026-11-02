"""Failure-path and CLI regression tests for the research-only V6 bridge."""
import io
import json
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import v6_three_model_qaoa_bridge as bridge


class V6BridgeFailures(unittest.TestCase):
    def setUp(self):
        self.archive = Path(bridge.__file__).with_name("historical_archive.csv")
        self.base = bridge.run(self.archive)

    def invalid_prediction(self, replacement):
        from modular_selection_pipeline import run as original
        def altered(draws):
            result = original(draws)
            result["predictions"]["main_5_10"].update(replacement)
            return result
        return altered

    def test_invalid_first_zone_is_rejected(self):
        for first in ([1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 39]):
            with self.subTest(first=first):
                with patch.object(bridge, "run_three", side_effect=self.invalid_prediction({"first": first})):
                    with self.assertRaisesRegex(ValueError, "Invalid model first-zone"):
                        bridge.run(self.archive)

    def test_invalid_second_zone_is_rejected(self):
        for second in (0, 9, "1"):
            with self.subTest(second=second):
                with patch.object(bridge, "run_three", side_effect=self.invalid_prediction({"second": second})):
                    with self.assertRaisesRegex(ValueError, "Invalid second-zone"):
                        bridge.run(self.archive)

    def test_qaoa_draw_cutoff_mismatch_fails_closed(self):
        from v6_ai_qaoa_e2e import run as original
        def altered(path):
            result = original(path)
            result["last_draw"] = "000000000"
            return result
        with patch.object(bridge, "run_qaoa", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "cutoffs disagree"):
                bridge.run(self.archive)

    def test_cli_outputs_valid_json_and_honest_flags(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            runpy.run_path(str(Path(bridge.__file__)), run_name="__main__")
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["cutoff"], self.base["cutoff"])
        self.assertFalse(result["full_e2e_verified"])
        self.assertFalse(result["live_official_api_verified"])
        self.assertTrue(result["production_selection_unchanged"])


if __name__ == "__main__":
    unittest.main()
