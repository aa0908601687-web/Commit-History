import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("wp_official_updater", Path(__file__).with_name("update_official_data.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class OfficialDataTests(unittest.TestCase):
    def test_accepts_valid_schema(self):
        payload = {"content": {"superLotto638Res": [
            {"period": "115000084", "drawNumberSize": [9, 10, 21, 24, 27, 36, 8]}
        ]}}
        self.assertEqual(module.parse_response(payload)[0], ("115000084", [9, 10, 21, 24, 27, 36], 8))

    def test_rejects_bad_numbers(self):
        with self.assertRaises(ValueError):
            module.parse_response({"content": {"superLotto638Res": [
                {"period": "115000084", "drawNumberSize": [9, 9, 21, 24, 27, 36, 8]}
            ]}})

    def test_rejects_changed_schema(self):
        with self.assertRaises(ValueError):
            module.parse_response({"content": {"superLotto638Res": [{"period": "115000084"}]}})

if __name__ == "__main__":
    unittest.main()
