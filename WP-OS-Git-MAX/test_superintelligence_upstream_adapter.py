import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("superintelligence_upstream_adapter.py")
SPEC = importlib.util.spec_from_file_location("superintelligence_upstream_adapter", MODULE_PATH)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter)


class SuperintelligenceUpstreamAdapterTests(unittest.TestCase):
    def _layout(self, root: Path) -> None:
        for name in adapter.REQUIRED_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".py":
                path.write_text("VALUE = 1\n", encoding="utf-8")
            else:
                path.write_text("MIT License\n", encoding="utf-8")

    def test_complete_non_git_layout_is_not_a_pinned_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            result = adapter.validate(root)
            self.assertTrue(result["structure_ok"])
            self.assertIsNone(result["commit_matches"])
            self.assertFalse(result["pinned_checkout_ok"])
            self.assertEqual(result["execution_status"], "not_executed")
            self.assertFalse(result["prediction_integration"])

    def test_pinned_layout_is_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            with mock.patch.object(adapter, "_git_head", return_value=adapter.PINNED_COMMIT):
                result = adapter.validate(root)
            self.assertTrue(result["pinned_checkout_ok"])
            self.assertEqual(result["connection_status"], "provenance_connected")
            self.assertEqual(result["observed_commit"], adapter.PINNED_COMMIT)
            self.assertTrue(result["file_sha256"])

    def test_wrong_commit_fails_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            with mock.patch.object(adapter, "_git_head", return_value="0" * 40):
                result = adapter.validate(root)
            self.assertFalse(result["commit_matches"])
            self.assertEqual(result["connection_status"], "failed")

    def test_missing_file_fails_structure_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            (root / "models/auto_ml.py").unlink()
            result = adapter.validate(root)
            self.assertFalse(result["structure_ok"])
            self.assertIn("models/auto_ml.py", result["missing_files"])

    def test_python_syntax_error_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            (root / "main.py").write_text("if :\n", encoding="utf-8")
            result = adapter.validate(root)
            self.assertFalse(result["structure_ok"])
            self.assertIn("main.py", result["syntax_errors"])

    def test_cli_writes_success_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evidence" / "result.json"
            self._layout(root)
            argv = [
                "adapter",
                "--upstream-dir", str(root),
                "--output", str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(adapter, "_git_head", return_value=adapter.PINNED_COMMIT),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                code = adapter.main()
            self.assertEqual(code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["pinned_checkout_ok"])
            self.assertEqual(json.loads(stdout.getvalue()), payload)

    def test_cli_returns_failure_for_unpinned_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            argv = ["adapter", "--upstream-dir", str(root)]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(adapter, "_git_head", return_value=None),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                code = adapter.main()
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(stdout.getvalue())["pinned_checkout_ok"])


if __name__ == "__main__":
    unittest.main()
