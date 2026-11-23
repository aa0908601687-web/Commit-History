"""Strict, fail-closed internal integrity gate for OF-LGBM historical CSV files.

Validates bytes, schema, chronological structure and manifest consistency.
Official authenticity requires an independent source comparison, NOT this gate.
"""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "verification" / "of_lgbm_data"
MANIFEST = DATA / "SOURCE_MANIFEST.json"
sys.path.insert(0, str(ROOT / "verification"))
from of_lgbm_real_history_gate import audit_csv

EXPECTED_GAME_FILES = {
    "power": "power_history_archive.csv",
    "lotto": "lotto_history_mirror.csv",
    "539": "539_history_mirror.csv",
}


def verify_manifest(metadata, data_root=DATA):
    """Check exact CSV bytes and chronological content against pinned manifest.

    Any CSV update must intentionally update the digest AND be independently
    reviewed; a matching digest is not proof of official origin.
    """
    if set(metadata["datasets"]) != set(EXPECTED_GAME_FILES):
        raise ValueError("unexpected dataset set")
    if metadata["record_by_record_officially_authenticated"] is not False:
        raise ValueError("official provenance cannot be inferred from checksums")
    if metadata["integrity_audit_20261119"]["official_files_imported"] is not False:
        raise ValueError("unsupported official source attestation")
    results = {}
    for game, filename in EXPECTED_GAME_FILES.items():
        item = metadata["datasets"][game]
        path = data_root / filename
        if item["path"] != "verification/of_lgbm_data/" + filename:
            raise ValueError("unexpected dataset path: " + game)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("SHA-256 mismatch for " + game)
        result = audit_csv(path, game)
        expected_last = item.get("last", item.get("cutoff"))
        if result["rows"] != item["rows"] or result["last_id"] != expected_last:
            raise ValueError("row count or cutoff mismatch for " + game)
        if game == "power":
            if result["date_coverage"] != item["rows"] or item["date_status"] != "all_1964_dates_restored_original_source_crosschecked":
                raise ValueError("Power dates must be complete and explicitly release-unapproved")
            if item["official_row_verification"] != "cross_checked_not_release_approved":
                raise ValueError("Power independent source review has not approved official release")
            if metadata.get("power_date_backfill_20261124", {}).get("official_source_release_approval") is not False:
                raise ValueError("unsupported official source release approval")
        else:
            if result["date_coverage"] != item["rows"]:
                raise ValueError("partial date coverage for " + game)
            if item["date_status"] != "format_consistent_not_officially_verified":
                raise ValueError("unsubstantiated date status for " + game)
        if game != "power" and item["official_row_verification"] != "not_completed":
            raise ValueError("unsupported provenance claim for " + game)
        results[game] = {"rows": result["rows"], "sha256": digest,
                         "cutoff": result["last_id"], "date_coverage": result["date_coverage"]}
    return results


class ManifestIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_three_csvs_match_pinned_manifest(self):
        result = verify_manifest(self.metadata)
        self.assertEqual(set(result), {"power", "lotto", "539"})
        self.assertEqual(result["power"]["date_coverage"], 1964)

    def test_date_gaps_and_official_provenance_not_misrepresented(self):
        self.assertFalse(self.metadata["record_by_record_officially_authenticated"])
        self.assertTrue(self.metadata["update_notice_20261119"]["power_dates_missing"])
        self.assertFalse(self.metadata["update_notice_20261119"]["independent_prospective_edge_verified"])
        self.assertEqual(self.metadata["integrity_audit_20261119"]["missing_power_draw_dates"],
                         self.metadata["power_date_backfill_20261124"]["rows"])
        self.assertEqual(self.metadata["power_date_backfill_20261124"]["missing_dates"], 0)
        self.assertFalse(self.metadata["power_date_backfill_20261124"]["official_source_release_approval"])

    def test_wrong_sha_is_rejected(self):
        data = json.loads(json.dumps(self.metadata))
        data["datasets"]["539"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_manifest(data)

    def test_fake_official_claim_is_rejected(self):
        data = json.loads(json.dumps(self.metadata))
        data["record_by_record_officially_authenticated"] = True
        with self.assertRaisesRegex(ValueError, "official provenance"):
            verify_manifest(data)

    def test_missing_row_and_cutoff_mismatch_is_rejected(self):
        data = json.loads(json.dumps(self.metadata))
        data["datasets"]["lotto"]["rows"] += 1
        with self.assertRaisesRegex(ValueError, "row count or cutoff"):
            verify_manifest(data)

    def test_source_path_swaps_are_rejected(self):
        data = json.loads(json.dumps(self.metadata))
        data["datasets"]["power"]["path"] = "verification/of_lgbm_data/lotto_history_mirror.csv"
        with self.assertRaisesRegex(ValueError, "unexpected dataset path"):
            verify_manifest(data)


if __name__ == "__main__":
    unittest.main()
