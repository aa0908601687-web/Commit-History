import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PackagingTests(unittest.TestCase):
    def test_scripts_compile(self):
        for name in ("verify_install.py", "smoke_predict.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"))

    def test_requirements_include_official_sdk(self):
        requirements = (ROOT / "requirements.txt").read_text()
        self.assertIn("structured-data-models", requirements)
        self.assertIn("scikit-learn", requirements)

    def test_smoke_has_actual_inference(self):
        content = (ROOT / "smoke_predict.py").read_text()
        self.assertIn("KumoTabular(", content)
        self.assertIn("x_query=", content)
        self.assertIn("num_estimators=", content)

    def test_import_verification_writes_evidence(self):
        content = (ROOT / "verify_install.py").read_text()
        self.assertIn("installation_check.json", content)

if __name__ == "__main__":
    unittest.main()
