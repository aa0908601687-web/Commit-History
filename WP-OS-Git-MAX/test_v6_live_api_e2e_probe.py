"""Offline adversarial checks; these do NOT prove live endpoint connectivity."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import csv
import io
import json
from contextlib import redirect_stdout

import v6_live_api_e2e_probe as probe


class LiveProbeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name) / "archive.csv"
        self.records = [(f"115000{i:03d}", [1, 3, 7, 12, 20, 38], 4) for i in range(1, 22)]
        with self.archive.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["draw_id", "first", "second"])
            for k, nums, second in self.records:
                writer.writerow([k, " ".join(map(str, nums)), second])

    def fetch(self, month):
        return self.records[-4:] if month == "2026-10" else [("115000022", [2, 4, 9, 11, 20, 37], 5)]

    def bridge(self, path):
        import hashlib
        with open(path, newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        return {"cutoff": records[-1]["draw_id"], "source_archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    def test_success_uses_temp_file_and_preserves_archive(self):
        before = self.archive.read_bytes()
        result = probe.probe("2026-11", self.archive, self.fetch, self.bridge)
        self.assertEqual(result["merged_cutoff"], "115000022")
        self.assertEqual(result["status"], "new_draws_imported_and_verified")
        self.assertEqual(result["new_draw_count"], 1)
        self.assertEqual(result["new_draw_periods"], ["115000022"])
        self.assertEqual(result["overlap_verified"], 4)
        self.assertEqual(result["months_requested"], ("2026-10", "2026-11"))
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertFalse(result["trained_ai_verified"])

    def test_conflicting_period_fails_closed(self):
        def bad(month):
            return [(self.records[-1][0], [2, 3, 4, 5, 6, 7], 1)]
        with self.assertRaisesRegex(ValueError, "API/archive conflict"):
            probe.probe("2026-11", self.archive, bad, self.bridge)

    def test_no_overlap_rejected(self):
        def unrelated(month):
            return [("115000080", [2, 3, 4, 5, 6, 7], 1)]
        with self.assertRaisesRegex(ValueError, "No shared"):
            probe.probe("2026-11", self.archive, unrelated, self.bridge)

    def test_bridge_cutoff_mismatch_fails(self):
        def wrong(path):
            result = self.bridge(path)
            result["cutoff"] = "115000001"
            return result
        with self.assertRaisesRegex(ValueError, "cutoff differs"):
            probe.probe("2026-11", self.archive, self.fetch, wrong)

    def test_bridge_hash_mismatch_fails(self):
        def wrong(path):
            result = self.bridge(path)
            result["source_archive_sha256"] = "invalid"
            return result
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            probe.probe("2026-11", self.archive, self.fetch, wrong)

    def test_no_network_fallback_after_error(self):
        def fail(month):
            raise ConnectionError("API unavailable")
        with self.assertRaisesRegex(ConnectionError, "unavailable"):
            probe.probe("2026-11", self.archive, fail, self.bridge)

    def test_empty_current_month_is_explicit_not_falsely_updated(self):
        def empty_current(month):
            if month == "2026-11":
                raise ValueError("No official draws returned")
            return self.records[-4:]
        result = probe.probe("2026-11", self.archive, empty_current, self.bridge)
        self.assertEqual(result["months_without_published_draws"], ["2026-11"])
        self.assertEqual(result["merged_cutoff"], self.records[-1][0])
        self.assertEqual(result["api_rows"], 4)
        self.assertEqual(result["status"], "no_new_draws_live_bridge_verified")
        self.assertEqual(result["new_draw_count"], 0)
        self.assertFalse(result["latest_draw_advanced"])

    def test_empty_previous_month_does_not_silently_pass(self):
        def empty_any(month):
            raise ValueError("No official draws returned")
        with self.assertRaisesRegex(ValueError, "No official draws"):
            probe.probe("2026-11", self.archive, empty_any, self.bridge)

    def test_088_mocked_new_draw_is_never_written_to_archive(self):
        """Fixture verifies the 088 merge path, NOT official API publication."""
        original = self.archive.read_bytes()
        last = ("115000088", [4, 6, 16, 19, 21, 37], 7)
        def mocked_fetch(month):
            if month == "2026-10":
                return self.records[-4:]
            return [last]
        result = probe.probe("2026-11", self.archive, mocked_fetch, self.bridge)
        self.assertEqual(result["new_draw_periods"], ["115000088"])
        self.assertEqual(result["merged_cutoff"], "115000088")
        self.assertEqual(result["new_draw_count"], 1)
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertFalse(result["trained_ai_verified"])

    def test_duplicate_identical_api_records_are_deduplicated(self):
        def duplicated(month):
            return self.records[-4:] if month == "2026-10" else [self.records[-1]]
        result = probe.probe("2026-11", self.archive, duplicated, self.bridge)
        self.assertEqual(result["api_rows"], 5)
        self.assertEqual(result["api_distinct_periods"], 4)
        self.assertEqual(result["new_draw_count"], 0)

    def test_previous_month_year_boundary(self):
        self.assertEqual(probe.previous_month("2026-01"), "2025-12")
        self.assertEqual(probe.previous_month("2026-11"), "2026-10")

    def test_conflicting_api_rows_rejected(self):
        def contradictory(month):
            if month == "2026-10":
                return self.records[-4:]
            return [(self.records[-1][0], [2, 4, 9, 11, 20, 37], 5)]
        with self.assertRaisesRegex(ValueError, "Contradictory API records"):
            probe.probe("2026-11", self.archive, contradictory, self.bridge)

    def test_cli_output_is_json_with_mocked_probe(self):
        from unittest.mock import patch
        output = io.StringIO()
        with patch.object(probe, "probe", return_value={"status": "fixture_only"}), \
             patch("sys.argv", ["v6_live_api_e2e_probe.py"]), \
             redirect_stdout(output):
            probe.main()
        self.assertEqual(json.loads(output.getvalue()), {"status": "fixture_only"})

    def test_cli_writes_output_file(self):
        target = Path(self.tmp.name) / "report.json"
        with patch.object(probe, "probe", return_value={"status": "fixture_only"}), \
             patch("sys.argv", ["v6_live_api_e2e_probe.py", "--month", "2026-11", "--output", str(target)]), \
             redirect_stdout(io.StringIO()):
            probe.main()
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "fixture_only"})

    def test_invalid_month_rejected(self):
        with self.assertRaises(ValueError):
            probe.probe("2026-13", self.archive, self.fetch, self.bridge)


if __name__ == "__main__":
    unittest.main()
