"""Defensive tests of the three-game local snapshot completeness report."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_history_completeness import audit_all

DATA=Path(__file__).resolve().parent/"of_lgbm_data"
MANIFEST=DATA/"SOURCE_MANIFEST.json"

class HistoryCompletenessTests(unittest.TestCase):
    def test_all_three_games_and_missing_dates_reported(self):
        r=audit_all(DATA, MANIFEST)
        self.assertEqual(r["dataset_count"],3)
        self.assertEqual(set(r["datasets"]),{"power","lotto","539"})
        self.assertEqual(r["datasets"]["power"]["missing_draw_dates"],1964)
        self.assertEqual(r["datasets"]["power"]["date_coverage"],0)
        self.assertEqual(r["datasets"]["lotto"]["missing_draw_dates"],0)
        self.assertEqual(r["datasets"]["539"]["missing_draw_dates"],0)
        self.assertFalse(r["official_full_history_verified"])
        self.assertFalse(r["complete_to_today_verified"])
        self.assertTrue(all(not d["official_record_by_record_verified"]
                            and not d["full_historical_coverage_certified"]
                            for d in r["datasets"].values()))

    def test_false_official_claim_rejected(self):
        d=json.loads(MANIFEST.read_text())
        d["record_by_record_officially_authenticated"]=True
        with tempfile.TemporaryDirectory() as td:
            m=Path(td)/"manifest.json";m.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"official provenance"):
                audit_all(DATA,m)

    def test_corrupted_checksum_rejected(self):
        d=json.loads(MANIFEST.read_text())
        d["datasets"]["power"]["sha256"]="0"*64
        with tempfile.TemporaryDirectory() as td:
            m=Path(td)/"manifest.json";m.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                audit_all(DATA,m)

    def test_fabricated_date_completion_flag_rejected(self):
        d=json.loads(MANIFEST.read_text())
        d["datasets"]["power"]["date_status"]="complete_official"
        with tempfile.TemporaryDirectory() as td:
            m=Path(td)/"manifest.json";m.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"date provenance"):
                audit_all(DATA,m)

    def test_tampered_date_gap_tally_rejected(self):
        d=json.loads(MANIFEST.read_text())
        d["integrity_audit_20261119"]["missing_power_draw_dates"]=0
        with tempfile.TemporaryDirectory() as td:
            m=Path(td)/"manifest.json";m.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"missing-date tally"):
                audit_all(DATA,m)

if __name__=="__main__":
    unittest.main()
