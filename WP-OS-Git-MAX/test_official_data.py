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

    def test_archive_retains_rows_and_conflicts_do_not_overwrite(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "latest.csv"
            history = Path(directory) / "archive.csv"
            header = "draw_id,first,second\n"
            history.write_text(header + "".join(f"{115000001+i},01 02 03 04 05 06,01\n" for i in range(11)))
            data.write_text(header + "".join(f"{115000001+i},01 02 03 04 05 06,01\n" for i in range(1,11)))
            with patch.object(module, "DATA", data), patch.object(module, "HISTORY", history), patch.object(module, "fetch", return_value=[("115000012", [1,2,3,4,5,6], 1)]), patch("sys.argv", ["updater"]):
                module.main()
                self.assertEqual(len(module.existing(history)), 12)
                self.assertEqual(len(module.existing(data)), 10)
                before = (data.read_bytes(), history.read_bytes())
                with patch.object(module, "fetch", return_value=[("115000012", [1,2,3,4,5,7], 1)]):
                    with self.assertRaises(ValueError):
                        module.main()
                self.assertEqual(before, (data.read_bytes(), history.read_bytes()))

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

