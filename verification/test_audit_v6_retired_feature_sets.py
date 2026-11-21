import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("audit_v6_retired_feature_sets.py")
spec = importlib.util.spec_from_file_location("audit_v6_retired_feature_sets", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class RetiredFeatureSuiteAuditTests(unittest.TestCase):
    def test_active_paths_are_clear(self):
        self.assertEqual(module.audit(), [])

    def test_retired_names_are_detected(self):
        for value in ("100_feature_sets", "100 candidate feature combinations",
                      "feature_suites_100", "feature combinations = 100"):
            with self.subTest(value=value):
                self.assertRegex(value, module.RETIRED)

    def test_unrelated_100_values_are_allowed(self):
        for value in ("OpenFE 100% enabled", "100 historical draws",
                      "max_features=12", "candidate pool of 8"):
            with self.subTest(value=value):
                self.assertNotRegex(value, module.RETIRED)

if __name__ == "__main__":
    unittest.main()
