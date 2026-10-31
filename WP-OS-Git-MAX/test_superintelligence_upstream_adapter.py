import importlib.util
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
