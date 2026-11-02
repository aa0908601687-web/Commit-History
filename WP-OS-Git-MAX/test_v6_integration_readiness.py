"""Tests for the V6 fail-closed integration readiness audit."""
import copy
import io
import json
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import v6_integration_readiness as readiness


class ReadinessTests(unittest.TestCase):
    def test_archive_parity_and_honest_status(self):
        report = readiness.check()
        self.assertGreaterEqual(report["data"]["archive_count"], 20)
        self.assertEqual(report["three_model_parity"], "pass")
        self.assertFalse(report["full_e2e"])
        self.assertIn("not checked", report["official_api"])
        self.assertEqual(len(report["data"]["sha256"]), 64)

    def test_insufficient_history_rejected(self):
        with patch("wp_os_git_max.load", return_value=[{"draw_id": "115000001"}]):
            with self.assertRaisesRegex(ValueError, "20 stored"):
                readiness.check()

    def test_mismatching_prediction_is_rejected(self):
        from wp_os_git_max import predict
        def different(draws):
            result = copy.deepcopy(predict(draws))
            result["through_draw"] = "000000000"
            return result
        with patch("wp_os_git_max.predict", side_effect=different):
            with self.assertRaisesRegex(AssertionError, "parity/cutoff mismatch"):
                readiness.check()

    def test_cli_prints_json(self):
        out = io.StringIO()
        with patch.object(sys, "argv", ["v6-readiness"]), redirect_stdout(out):
            readiness.main()
        self.assertFalse(json.loads(out.getvalue())["full_e2e"])

    def test_strict_cli_fails_closed(self):
        with patch.object(sys, "argv", ["v6-readiness", "--strict-v6"]), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "NOT READY"):
                readiness.main()

    def test_entrypoint(self):
        with patch.object(sys, "argv", ["v6-readiness"]), redirect_stdout(io.StringIO()):
            runpy.run_path(str(Path(readiness.__file__)), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
