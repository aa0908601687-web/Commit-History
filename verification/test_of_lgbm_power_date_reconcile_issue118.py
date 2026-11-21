"""Regression tests for fail-closed Power date reconciliation (Issue #118)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from verification.of_lgbm_power_date_reconcile import reconcile


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


class PowerDateReconcileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.archive = root / "archive.csv"
        self.source = root / "official_import.csv"
        self.output = root / "dated.csv"
        self.report = root / "audit.json"
        self.first = "01 02 03 04 05 06"
        write_csv(self.archive, ["draw_id", "first", "second"], [
            ["115000001", self.first, "01"],
            ["115000002", self.first, "02"],
        ])

    def source_rows(self, rows):
        write_csv(self.source, ["draw_id", "draw_date", "first", "second"], rows)

    def test_valid_structural_import_is_not_official_authentication(self):
        self.source_rows([
            ["115000001", "2026-01-01", self.first, "01"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        audit = reconcile(self.archive, self.source, self.output, self.report)
        self.assertEqual(audit["dated_rows"], 2)
        self.assertFalse(audit["independent_official_provenance_verified"])
        self.assertEqual(json.loads(self.report.read_text())["rows"], 2)

    def test_missing_draw_fails_without_outputs(self):
        self.source_rows([["115000001", "2026-01-01", self.first, "01"]])
        with self.assertRaisesRegex(ValueError, "source draw IDs differ"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_number_mismatch_fails_without_outputs(self):
        self.source_rows([
            ["115000001", "2026-01-01", "01 02 03 04 05 07", "01"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        with self.assertRaisesRegex(ValueError, "drawn numbers differ"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())

    def test_invalid_date_fails_without_outputs(self):
        self.source_rows([
            ["115000001", "2026-02-30", self.first, "01"],
            ["115000002", "2026-03-02", self.first, "02"],
        ])
        with self.assertRaisesRegex(ValueError, "invalid record"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())

    def test_refuses_overwriting_existing_result(self):
        self.source_rows([
            ["115000001", "2026-01-01", self.first, "01"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        self.output.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertEqual(self.output.read_text(), "keep")


    def test_refuses_overwriting_existing_audit_report(self):
        self.source_rows([
            ["115000001", "2026-01-01", self.first, "01"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        self.report.write_text("keep audit", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertEqual(self.report.read_text(encoding="utf-8"), "keep audit")
        self.assertFalse(self.output.exists())

    def test_rejects_reversed_date_order_without_outputs(self):
        self.source_rows([
            ["115000001", "2026-01-05", self.first, "01"],
            ["115000002", "2026-01-01", self.first, "02"],
        ])
        with self.assertRaisesRegex(ValueError, "dates not strictly increasing"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())


    def test_duplicate_source_draw_id_fails_closed(self):
        self.source_rows([
            ["115000001", "2026-01-01", self.first, "01"],
            ["115000001", "2026-01-05", self.first, "01"],
        ])
        with self.assertRaisesRegex(ValueError, "duplicate or malformed draw id"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_roc_year_mismatch_fails_closed(self):
        self.source_rows([
            ["115000001", "2025-01-01", self.first, "01"],
            ["115000002", "2025-01-05", self.first, "02"],
        ])
        with self.assertRaisesRegex(ValueError, "invalid ISO date/ROC year"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())


    def test_missing_required_source_column_fails_closed(self):
        write_csv(self.source, ["draw_id", "draw_date", "first"], [
            ["115000001", "2026-01-01", self.first],
            ["115000002", "2026-01-05", self.first],
        ])
        with self.assertRaisesRegex(ValueError, "missing mandatory columns"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_second_zone_mismatch_fails_closed(self):
        self.source_rows([
            ["115000001", "2026-01-01", self.first, "02"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        with self.assertRaisesRegex(ValueError, "drawn numbers differ"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())


    def test_invalid_first_zone_fails_without_outputs(self):
        self.source_rows([
            ["115000001", "2026-01-01", "01 01 03 04 05 06", "01"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        with self.assertRaisesRegex(ValueError, "invalid Power number"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_conflicting_source_never_mutates_inputs(self):
        self.source_rows([
            ["115000001", "2026-01-01", self.first, "02"],
            ["115000002", "2026-01-05", self.first, "02"],
        ])
        archive_before = self.archive.read_bytes()
        source_before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "drawn numbers differ"):
            reconcile(self.archive, self.source, self.output, self.report)
        self.assertEqual(self.archive.read_bytes(), archive_before)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

if __name__ == "__main__":
    unittest.main()
