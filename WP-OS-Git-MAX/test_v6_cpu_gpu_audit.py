"""Verify that GPU unavailability is never misreported as GPU PASS."""
import unittest
import runpy
import json
import tempfile
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_cpu_gpu_audit import verify

class Tensor:
    def __init__(self, xs):
        self.xs = list(xs)
    def square(self):
        return Tensor([x*x for x in self.xs])
    def __truediv__(self, n):
        return Tensor([x/n for x in self.xs])
    def __add__(self, b):
        return Tensor([a+c for a,c in zip(self.xs,b.xs)])
    def cpu(self):
        return self
    def tolist(self):
        return self.xs

class Cuda:
    def __init__(self, available):
        self.available = available
    def is_available(self):
        return self.available
    def get_device_name(self, i):
        return "FAKE GPU: TEST ONLY"

class Torch:
    __version__ = "fake-test-only"
    float64 = object()
    def __init__(self, cuda):
        self.cuda = Cuda(cuda)
    def arange(self, a, b, dtype, device):
        assert device in ("cpu","cuda:0")
        return Tensor(range(a,b))

class TestDeviceAudit(unittest.TestCase):
    def test_cpu_and_explicit_gpu_not_tested(self):
        r = verify(Torch(False))
        self.assertEqual(r["cpu"]["combinations"],28)
        self.assertEqual(r["gpu"]["status"],"NOT_TESTED")
    def test_required_gpu_unavailable_is_error(self):
        with self.assertRaisesRegex(RuntimeError,"GPU verification requested"):
            verify(Torch(False),require_gpu=True)
    def test_matching_device_path_with_fake_backend(self):
        r = verify(Torch(True),require_gpu=True)
        self.assertEqual(r["gpu"]["status"],"PASS")
        self.assertTrue(r["gpu"]["same_selection"])

    def test_gpu_numeric_disagreement_fails_closed(self):
        class BadGpuTorch(Torch):
            def arange(self, a, b, dtype, device):
                if device == "cuda:0":
                    return Tensor(range(b-1, a-1, -1))
                return super().arange(a, b, dtype, device)
        with self.assertRaisesRegex(AssertionError,"CPU/GPU result differs"):
            verify(BadGpuTorch(True),require_gpu=True)

    def test_command_line_writes_explicit_cpu_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "evidence.json")
            script = str(Path(__file__).resolve().parent / "v6_cpu_gpu_audit.py")
            with patch.dict(sys.modules, {"torch": Torch(False)}):
                with patch.object(sys, "argv", [script, "--output", output]):
                    with redirect_stdout(StringIO()):
                        runpy.run_path(script, run_name="__main__")
            report = json.loads(Path(output).read_text(encoding="utf-8"))
            self.assertEqual(report["cpu"]["status"], "PASS")
            self.assertEqual(report["cpu"]["combinations"], 28)
            self.assertEqual(report["gpu"]["status"], "NOT_TESTED")

if __name__ == "__main__":
    unittest.main()
