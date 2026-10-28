import io
import json
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from abe_model_connectivity import check_model

class DummyResponse:
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def read(self, *args): return b''
    def __iter__(self): return iter([])

class ConnectivityTests(unittest.TestCase):
    def test_mock_registry_success(self):
        class Response(DummyResponse):
            def read(self, *args): return b'{"id":"amazon/chronos-2","sha":"123abc"}'
        r = check_model("amazon/chronos-2", opener=lambda req, timeout: Response())
        self.assertTrue(r["registry_reachable"])
        self.assertFalse(r["inference_verified"])
    def test_registry_wrong_model_rejected(self):
        class Response(DummyResponse):
            def read(self, *args): return b'{"id":"different/model"}'
        r = check_model("amazon/chronos-2", opener=lambda req, timeout: Response())
        self.assertFalse(r["registry_reachable"])
    def test_registry_connection_error(self):
        def fail(req, timeout): raise OSError("offline")
        r = check_model("amazon/chronos-2", opener=fail)
        self.assertFalse(r["registry_reachable"])
        self.assertFalse(r["inference_verified"])

if __name__ == "__main__":
    unittest.main()
