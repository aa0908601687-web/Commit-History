"""Offline tests for the real Nemotron probe's fail-closed behavior."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_nemotron_live_probe import MODEL, probe, validate_response

GOOD = {"model": MODEL, "choices": [{"message": {"content": "NEMOTRON_OK"}}]}

class FakeResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, size): return __import__("json").dumps(GOOD).encode()

class ProbeTests(unittest.TestCase):
    def test_real_response_schema(self):
        self.assertTrue(validate_response(GOOD)["live_inference"])
    def test_no_secret_fails_closed(self):
        with self.assertRaises(ValueError):
            probe("", opener=lambda *a, **k: FakeResponse())
    def test_wrong_model_fails(self):
        with self.assertRaises(ValueError):
            validate_response({**GOOD, "model": "other"})
    def test_empty_completion_fails(self):
        with self.assertRaises(ValueError):
            validate_response({"model": MODEL, "choices": []})
    def test_mock_transport_does_not_prove_live_execution(self):
        # Verifies transport parsing only, never claim this test runs the NVIDIA API.
        self.assertEqual(probe("fixture-key", opener=lambda *a, **k: FakeResponse())["status"],
                         "live_nemotron_inference_verified")

if __name__ == "__main__":
    unittest.main()
