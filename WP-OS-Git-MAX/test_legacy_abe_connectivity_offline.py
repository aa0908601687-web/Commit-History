"""Legacy connectivity tests only. No network access, no V6/ABE integration."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import abe_model_connectivity as legacy


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self, size=-1):
        return json.dumps(self.payload).encode()


class LegacyConnectivityOfflineTests(unittest.TestCase):
    def test_registry_success_only_means_reachable(self):
        def opener(request, timeout):
            self.assertEqual(timeout, 3)
            self.assertIn("huggingface.co/api/models/", request.full_url)
            return FakeResponse({"id": "amazon/chronos-2", "sha": "test-revision"})
        result = legacy.check_model("amazon/chronos-2", opener=opener, timeout=3)
        self.assertTrue(result["registry_reachable"])
        self.assertFalse(result["inference_verified"])
        self.assertEqual(result["revision"], "test-revision")

    def test_identity_mismatch_fails_closed(self):
        result = legacy.check_model("amazon/chronos-2", opener=lambda *a, **kw:
                                    FakeResponse({"id": "incorrect"}))
        self.assertFalse(result["registry_reachable"])
        self.assertFalse(result["inference_verified"])

    def test_network_error_fails_closed(self):
        def fail(*args, **kwargs):
            raise OSError("offline")
        result = legacy.check_model("amazon/chronos-2", opener=fail)
        self.assertFalse(result["registry_reachable"])
        self.assertEqual(result["error"], "offline")

    def test_cli_report_success_and_explicit_non_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            with patch.object(legacy, "check_model", return_value={
                "registry_reachable": True, "inference_verified": False}), \
                 patch.object(sys, "argv", ["legacy", "--output", str(target)]), \
                 redirect_stdout(io.StringIO()) as printed:
                legacy.main()
            result = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(printed.getvalue()), result)
            self.assertTrue(result["all_registries_reachable"])
            self.assertFalse(result["full_five_model_inference_verified"])

    def test_cli_report_failure_exits_nonzero(self):
        with patch.object(legacy, "check_model", return_value={
            "registry_reachable": False, "inference_verified": False}), \
             patch.object(sys, "argv", ["legacy"]), \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                legacy.main()
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
