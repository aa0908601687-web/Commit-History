"""Consistency gate for OF-LGBM repository CSV snapshots and source manifest.

This validates internal evidence only; it does not certify official draw provenance.
"""
import csv
import json
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "verification" / "of_lgbm_data"
MANIFEST = DATA / "SOURCE_MANIFEST.json"
EXPECTED = {"power": ("power_history_archive.csv", "115000092", 1964),
            "lotto": ("lotto_history_mirror.csv", "115000106", 412),
            "539": ("539_history_mirror.csv", "115000279", 1119)}

class ManifestIntegrityTests(unittest.TestCase):
    def test_manifest_matches_all_csv_snapshot_boundaries(self):
        metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for game, (filename, expected_cutoff, expected_rows) in EXPECTED.items():
            with self.subTest(game=game):
                path = DATA / filename
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected_rows)
                self.assertEqual(rows[-1]["draw_id"], expected_cutoff)
                entry = metadata["datasets"][game]
                self.assertEqual(entry["rows"], len(rows))
                self.assertEqual(entry.get("last", entry.get("cutoff")), rows[-1]["draw_id"])
                self.assertFalse(metadata["record_by_record_officially_authenticated"])
                self.assertEqual(len(hashlib.sha256(path.read_bytes()).hexdigest()), 64)

    def test_power_missing_dates_not_marked_verified(self):
        metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with (DATA / "power_history_archive.csv").open(encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        self.assertNotIn("draw_date", header)
        self.assertTrue(metadata["update_notice_20261119"]["power_dates_missing"])

    def test_third_party_not_represented_as_official(self):
        metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertFalse(metadata["record_by_record_officially_authenticated"])
        self.assertFalse(metadata["update_notice_20261119"]["independent_prospective_edge_verified"])

if __name__ == "__main__":
    unittest.main()
