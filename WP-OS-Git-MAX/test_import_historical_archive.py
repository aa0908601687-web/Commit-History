"""Historical importer tests: merge, dry-run, conflict, validation."""
import csv
import json
import io
import contextlib
import runpy
from unittest.mock import patch
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_historical_archive import prepare, main

class ImportHistoryTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.folder = Path(self.work.name)
        self.archive = self.folder / "archive.csv"
        self.source = self.folder / "incoming.csv"
        self.archive.write_text("draw_id,first,second\n115000001,01 02 03 04 05 06,01\n")
        self.source.write_text("draw_id,first,second\n115000002,02 03 04 05 06 07,02\n")

    def test_merges_without_touching_original(self):
        before = self.archive.read_bytes()
        contents, info = prepare(self.archive, self.source)
        self.assertEqual(info["total"], 2)
        self.assertEqual(info["new_draws"], 1)
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertIn("115000002", contents)

    def test_repeated_import_idempotent(self):
        contents, _ = prepare(self.archive, self.source)
        self.archive.write_text(contents)
        _, report = prepare(self.archive, self.source)
        self.assertEqual(report["new_draws"], 0)

    def test_conflict_is_rejected(self):
        self.source.write_text("draw_id,first,second\n115000001,02 03 04 05 06 07,02\n")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            prepare(self.archive, self.source)

    def test_unexpected_csv_columns_rejected(self):
        self.source.write_text("period,numbers,bonus\n115000002,01 02 03 04 05 06,01\n")
        with self.assertRaisesRegex(ValueError, "Expected draw_id,first,second columns"):
            prepare(self.archive, self.source)

    def test_duplicate_rejected(self):
        self.source.write_text("draw_id,first,second\n115000002,01 02 03 04 05 06,01\n115000002,01 02 03 04 05 06,01\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare(self.archive, self.source)

    def test_invalid_numbers_rejected(self):
        self.source.write_text("draw_id,first,second\n115000002,01 02 03 04 05 39,01\n")
        with self.assertRaisesRegex(ValueError, "Invalid draw"):
            prepare(self.archive, self.source)

    def test_new_archive_and_cli_dry_run(self):
        missing = self.folder / "new.csv"
        data, report = prepare(missing, self.source)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["previous_count"], 0)
        self.assertFalse(missing.exists())
        output = io.StringIO()
        with patch.object(sys, "argv", ["import_historical_archive.py", "--source", str(self.source),
                                         "--archive", str(self.archive)]):
            with contextlib.redirect_stdout(output):
                main()
        self.assertFalse(json.loads(output.getvalue())["wrote"])
        self.assertEqual(len(self.archive.read_text().splitlines()), 2)

    def test_cli_write_updates_test_archive(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["import_historical_archive.py", "--source", str(self.source),
                                         "--archive", str(self.archive), "--write"]):
            with contextlib.redirect_stdout(output):
                main()
        self.assertTrue(json.loads(output.getvalue())["wrote"])
        self.assertEqual(len(self.archive.read_text().splitlines()), 3)

    def test_script_entrypoint(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["import_historical_archive.py", "--source", str(self.source),
                                         "--archive", str(self.archive)]):
            with contextlib.redirect_stdout(output):
                runpy.run_path(str(Path(__file__).resolve().parent / "import_historical_archive.py"), run_name="__main__")
        self.assertFalse(json.loads(output.getvalue())["wrote"])

if __name__ == "__main__":
    unittest.main()
