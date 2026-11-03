import sys
import types
import unittest
from unittest.mock import patch

import v6_cudaq_backend as backend


class _FakeState:
    def __getitem__(self, index):
        # A deterministic maximum at one valid six-bit mask.
        return 1.0 if index == 0b00111111 else 0.001


class _FakeCudaQ(types.SimpleNamespace):
    __version__ = "test"

    def __init__(self):
        super().__init__()
        self.target = None
        self.seed = None
        self.kernel = lambda function: function

    def set_target(self, target):
        self.target = target

    def set_random_seed(self, seed):
        self.seed = seed

    def get_state(self, *args):
        return _FakeState()


class CudaQBackendTests(unittest.TestCase):
    def setUp(self):
        self.pool = list(range(1, 9))
        self.scores = {number: number / 10 for number in self.pool}

    def test_optional_dependency_falls_back_explicitly(self):
        real_import = backend.importlib.import_module

        def missing(name):
            if name == "cudaq":
                raise ImportError("not installed")
            return real_import(name)

        with patch.object(backend.importlib, "import_module", side_effect=missing):
            selected, metadata = backend.select_six(self.pool, self.scores)
        self.assertIsNone(selected)
        self.assertFalse(metadata["executed"])
        self.assertFalse(metadata["qpu"])
        self.assertIn("fallback", metadata)

    def test_required_dependency_fails_closed(self):
        with patch.object(backend, "availability", return_value={
            "installed": False, "backend": "qpp-cpu", "qpu": False, "reason": "missing"
        }):
            with self.assertRaisesRegex(RuntimeError, "required but unavailable"):
                backend.select_six(self.pool, self.scores, required=True)

    def test_local_backend_executes_and_never_claims_qpu(self):
        fake = _FakeCudaQ()
        with patch.dict(sys.modules, {"cudaq": fake}):
            selected, metadata = backend.select_six(self.pool, self.scores, required=True)
        self.assertEqual(selected, [1, 2, 3, 4, 5, 6])
        self.assertTrue(metadata["executed"])
        self.assertEqual(metadata["backend"], "qpp-cpu")
        self.assertEqual(fake.target, "qpp-cpu")
        self.assertEqual(fake.seed, backend.SEED)
        self.assertFalse(metadata["real_qpu_connected"])
        self.assertFalse(metadata["remote_service_connected"])
        self.assertFalse(metadata["predictive_advantage_verified"])

    def test_rejects_invalid_pool_and_missing_score(self):
        with self.assertRaises(ValueError):
            backend.select_six([1] * 8, self.scores)
        with self.assertRaises(ValueError):
            backend.select_six(self.pool, {1: 0.1})


if __name__ == "__main__":
    unittest.main()
