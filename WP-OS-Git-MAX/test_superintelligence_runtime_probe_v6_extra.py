"""Offline, isolated tests of the optional research runtime probe."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import superintelligence_runtime_probe as runtime


class RuntimeProbeExtraTests(unittest.TestCase):
    def test_missing_upstream_module_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                runtime._load("missing_runtime_module", Path(directory) / "missing.py")

    def test_loader_rejects_spec_without_loader(self):
        with patch.object(runtime.importlib.util, "spec_from_file_location", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Cannot load upstream module"):
                runtime._load("invalid", Path("missing.py"))

    def test_main_success_writes_json_without_running_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "reports" / "probe.json"
            fixture = {"runtime_probe_passed": True, "prediction_integration": False}
            with patch.object(runtime, "probe", return_value=fixture), \
                 patch("sys.argv", ["probe", "--upstream-dir", directory, "--output", str(target)]), \
                 redirect_stdout(io.StringIO()) as output:
                code = runtime.main()
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(target.read_text()), fixture)
            self.assertEqual(json.loads(output.getvalue()), fixture)

    def test_main_rejects_unsuccessful_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runtime, "probe", return_value={"runtime_probe_passed": False}), \
                 patch("sys.argv", ["probe", "--upstream-dir", directory]), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.main(), 1)


if __name__ == "__main__":
    unittest.main()
