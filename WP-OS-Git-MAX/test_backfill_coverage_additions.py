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

    def empty_zip(self):
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w"):
            pass
        return bio.getvalue()

    def test_parse_and_render_valid_official_csv(self):
        raw = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\n威力彩,97000001,6,5,4,3,2,1,8\n"
        rows = self.module.parse_annual_zip(2008, self.zipped(raw))
        self.assertEqual(rows, [("097000001", [1, 2, 3, 4, 5, 6], 8)])
        self.assertIn("097000001,01 02 03 04 05 06,08", self.module.render(rows))

    def test_no_draws_or_empty_zip_rejected(self):
        with self.assertRaisesRegex(ValueError, "No 威力彩 records"):
            self.module.parse_annual_zip(2008, self.zipped("遊戲名稱,期別\n大樂透,097000001\n"))
        with self.assertRaisesRegex(ValueError, "ZIP is empty"):
            self.module.parse_annual_zip(2008, self.empty_zip())

    def test_missing_second_zone_and_bad_numbers_rejected(self):
        header = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\n"
        for tail, message in [
            ("1,2,3,4,5,6,", "lacks second-zone"),
            ("1,1,3,4,5,6,2", "Invalid first-zone"),
            ("1,2,3,4,5,6,9", "Invalid second-zone"),
        ]:
            with self.subTest(tail=tail):
                with self.assertRaisesRegex(ValueError, message):
                    self.module.parse_annual_zip(2008, self.zipped(header + "威力彩,097000001," + tail + "\n"))

    def test_archive_cache_requires_existing_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HISTORY_ARCHIVE_DIR": tmp}):
                with self.assertRaises(FileNotFoundError):
                    self.module.annual_download(2008)
                Path(tmp, "2008.zip").write_bytes(b"cached")
                url, payload = self.module.annual_download(2008)
                self.assertIn("2008.zip", url)
                self.assertEqual(payload, b"cached")


    def test_request_bytes_uses_response(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"test bytes"
        with mock.patch.object(self.module, "urlopen", return_value=response) as opening:
            self.assertEqual(self.module.request_bytes("https://example.invalid/x"), b"test bytes")
        self.assertEqual(opening.call_count, 1)

    def test_decode_csv_encodings_and_errors(self):
        self.assertEqual(self.module.decode_csv("威力彩".encode("cp950")), "威力彩")
        self.assertIsInstance(self.module.decode_csv(b"\\xff"), str)

    def test_parse_skips_invalid_headers_and_rows(self):
        header = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區"
        data = (header + "\n"
                "威力彩,097000001,01,02\n"
                "大樂透,097000001,01,02,03,04,05,06,02\n"
                "威力彩,097000001,01,02,03,04,05,06,02\n")
        self.assertEqual(len(self.module.parse_annual_zip(2008, self.zipped(data))), 1)

    def test_invalid_draw_id_rejected(self):
        header = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\n"
        with self.assertRaisesRegex(ValueError, "Invalid official period"):
            self.module.parse_annual_zip(2008, self.zipped(header + "威力彩,BAD,1,2,3,4,5,6,2\n"))

    def test_conflicting_annual_duplicate_rejected(self):
        header = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\n"
        data = header + ("威力彩,097000001,1,2,3,4,5,6,2\n"
                         "威力彩,097000001,1,2,3,4,5,7,2\n")
        with self.assertRaisesRegex(ValueError, "Conflicting official annual"):
            self.module.parse_annual_zip(2008, self.zipped(data))

    def test_main_rejects_annual_retrieval_failures_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.module, "ROOT", Path(tmp)), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", side_effect=OSError("offline")), \
                 mock.patch.object(self.module.time, "sleep") as sleeper:
                with self.assertRaisesRegex(RuntimeError, "inaccessible"):
                    self.module.main()
                self.assertEqual(sleeper.call_count, 3)
                self.assertFalse(list(Path(tmp).glob("*.csv")))

    def test_main_rejects_live_api_and_insufficient_history(self):
        one = ("115000001", [1, 2, 3, 4, 5, 6], 1)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.module, "ROOT", Path(tmp)), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", return_value=("official", b"data")), \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=[one]), \
                 mock.patch.object(self.module.time, "sleep"), \
                 mock.patch.object(self.module, "fetch", side_effect=OSError("offline")):
                with self.assertRaisesRegex(RuntimeError, "current month"):
                    self.module.main()
            with mock.patch.object(self.module, "ROOT", Path(tmp)), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", return_value=("official", b"data")), \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=[one]), \
                 mock.patch.object(self.module.time, "sleep"), \
                 mock.patch.object(self.module, "fetch", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    self.module.main()
            self.assertFalse(list(Path(tmp).glob("*.csv")))

    def test_main_rejects_conflicting_live_draw(self):
        one = ("115000001", [1, 2, 3, 4, 5, 6], 1)
        two = ("115000001", [1, 2, 3, 4, 5, 7], 1)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.module, "ROOT", Path(tmp)), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", return_value=("official", b"data")), \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=[one]), \
                 mock.patch.object(self.module.time, "sleep"), \
                 mock.patch.object(self.module, "fetch", return_value=[two]):
                with self.assertRaisesRegex(ValueError, "Annual/live conflict"):
                    self.module.main()


    def test_main_successful_atomic_backfill_and_manifest(self):
        import contextlib
        import json
        from io import StringIO
        first = [(f"{i:09d}", [1, 2, 3, 4, 5, 6], 1) for i in range(1, 1501)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(self.module, "ROOT", root), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", return_value=("fixture://2026", b"fake")), \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=first), \
                 mock.patch.object(self.module, "fetch", return_value=[]), \
                 mock.patch.object(self.module.time, "sleep"), \
                 contextlib.redirect_stdout(StringIO()) as output:
                self.module.main()
            archive = root / "historical_archive.csv"
            self.assertTrue(archive.exists())
            self.assertEqual((root / "historical_latest_20.csv").read_bytes(), archive.read_bytes())
            self.assertEqual(len((root / "historical_latest_10.csv").read_text().splitlines()), 11)
            manifest = json.loads((root / "history_backfill_manifest.json").read_text())
            self.assertEqual(manifest["draw_count"], 1500)
            self.assertEqual(json.loads(output.getvalue())["draw_count"], 1500)

    def test_main_existing_history_conflict_keeps_files_unchanged(self):
        first = [(f"{i:09d}", [1, 2, 3, 4, 5, 6], 1) for i in range(1, 1501)]
        old = ("000000001", [1, 2, 3, 4, 5, 7], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "historical_archive.csv").write_text("unchanged")
            with mock.patch.object(self.module, "ROOT", root), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[old]), \
                 mock.patch.object(self.module, "annual_download", return_value=("fixture://2026", b"fake")), \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=first), \
                 mock.patch.object(self.module, "fetch", return_value=[]), \
                 mock.patch.object(self.module.time, "sleep"):
                with self.assertRaisesRegex(ValueError, "not independently confirmed"):
                    self.module.main()
            self.assertEqual((root / "historical_archive.csv").read_text(), "unchanged")

    def test_main_annual_conflict_on_duplicate_years(self):
        first = ("000000001", [1, 2, 3, 4, 5, 6], 1)
        other = ("000000001", [1, 2, 3, 4, 5, 7], 1)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.module, "ROOT", Path(tmp)), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2025), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", return_value=("fixture", b"fake")), \
                 mock.patch.object(self.module, "parse_annual_zip", side_effect=[[first], [other]]), \
                 mock.patch.object(self.module.time, "sleep"):
                with self.assertRaisesRegex(ValueError, "Conflicting official draw"):
                    self.module.main()



    def test_official_parser_empty_csv_and_short_record(self):
        header = "遊戲名稱,期別,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,第二區\n"
        with self.assertRaisesRegex(ValueError, "No 威力彩 records"):
            self.module.parse_annual_zip(2008, self.zipped(""))
        with self.assertRaisesRegex(ValueError, "No 威力彩 records"):
            self.module.parse_annual_zip(2008, self.zipped(header + "威力彩,097000001,01\n"))

    def test_existing_history_missing_is_rejected_before_write(self):
        first = [(f"{i:09d}", [1, 2, 3, 4, 5, 6], 1) for i in range(1, 1501)]
        missing = ("999999999", [1, 2, 3, 4, 5, 6], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(self.module, "ROOT", root), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[missing]), \
                 mock.patch.object(self.module, "annual_download", return_value=("fixture", b"fake")), \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=first), \
                 mock.patch.object(self.module, "fetch", return_value=[]), \
                 mock.patch.object(self.module.time, "sleep"):
                with self.assertRaisesRegex(ValueError, "not independently confirmed"):
                    self.module.main()
            self.assertFalse((root / "historical_archive.csv").exists())



    def test_annual_download_without_local_cache(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.module, "request_bytes", return_value=b"downloaded") as get:
            url, data = self.module.annual_download(2026)
        self.assertTrue(url.endswith("/2026.zip"))
        self.assertEqual(data, b"downloaded")
        get.assert_called_once_with(url)

    def test_main_retries_then_rejects_incomplete_archive(self):
        record = ("115000001", [1, 2, 3, 4, 5, 6], 1)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.module, "ROOT", Path(tmp)), \
                 mock.patch.object(self.module, "FIRST_YEAR", 2026), \
                 mock.patch.object(self.module, "existing", return_value=[]), \
                 mock.patch.object(self.module, "annual_download", side_effect=[OSError("temporary"), ("fixture", b"x")]) as get, \
                 mock.patch.object(self.module, "parse_annual_zip", return_value=[record]), \
                 mock.patch.object(self.module, "fetch", return_value=[]), \
                 mock.patch.object(self.module.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    self.module.main()
                self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
