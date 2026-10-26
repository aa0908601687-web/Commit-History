"""Offline backfill parser and boundary tests for PR coverage; no external downloads."""
import importlib.util
import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "backfill_official_history.py"

@unittest.skipUnless(TARGET.exists(), "Backfill module only exists in merged main test checkout")
class BackfillCoverageAdditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TARGET.exists():
            return
        spec = importlib.util.spec_from_file_location("backfill_coverage", TARGET)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def zipped(self, csvtext):
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as z:
            z.writestr("draw.csv", csvtext)
        return bio.getvalue()

    def test_parse_and_render_valid_official_csv(self):
        raw = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\\n威力彩,97000001,6,5,4,3,2,1,8\\n"
        rows = self.module.parse_annual_zip(2008, self.zipped(raw))
        self.assertEqual(rows, [("097000001", [1, 2, 3, 4, 5, 6], 8)])
        self.assertIn("097000001,01 02 03 04 05 06,08", self.module.render(rows))

    def test_no_draws_or_empty_zip_rejected(self):
        with self.assertRaisesRegex(ValueError, "No 威力彩 records"):
            self.module.parse_annual_zip(2008, self.zipped("遊戲名稱,期別\\n大樂透,097000001\\n"))
        with self.assertRaisesRegex(ValueError, "ZIP is empty"):
            self.module.parse_annual_zip(2008, self.zipped(""))

    def test_missing_second_zone_and_bad_numbers_rejected(self):
        header = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\\n"
        for tail, message in [
            ("1,2,3,4,5,6,", "lacks second-zone"),
            ("1,1,3,4,5,6,2", "Invalid first-zone"),
            ("1,2,3,4,5,6,9", "Invalid second-zone"),
        ]:
            with self.subTest(tail=tail):
                with self.assertRaisesRegex(ValueError, message):
                    self.module.parse_annual_zip(2008, self.zipped(header + "威力彩,097000001," + tail + "\\n"))

    def test_archive_cache_requires_existing_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HISTORY_ARCHIVE_DIR": tmp}):
                with self.assertRaises(FileNotFoundError):
                    self.module.annual_download(2008)
                Path(tmp, "2008.zip").write_bytes(b"cached")
                url, payload = self.module.annual_download(2008)
                self.assertIn("2008.zip", url)
                self.assertEqual(payload, b"cached")

if __name__ == "__main__":
    unittest.main()
