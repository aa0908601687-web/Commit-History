"""OF-LGBM CSV gate validation tests; uses no external ML libraries."""
import csv
import sys
import tempfile
import unittest
from datetime import date,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from of_lgbm_real_history_gate import audit_csv

def fixture(folder,game):
    path=Path(folder)/"history.csv"
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["draw_id","draw_date","first"]+(["special"] if game=="lotto" else []))
        for i in range(20):
            fields=[f"115{i+1:06d}",(date(2026,1,1)+timedelta(days=i)).isoformat(),
                    "01 02 03 04 05"+(" 06" if game=="lotto" else "")]
            if game=="lotto": fields.append("07")
            writer.writerow(fields)
    return path

class HistoryGateTests(unittest.TestCase):
    def test_valid_history(self):
        for game in ("lotto","539"):
            with tempfile.TemporaryDirectory() as folder:
                report=audit_csv(fixture(folder,game),game)
                self.assertEqual(report["rows"],20)
                self.assertTrue(report["csv_dates_consistent"])
                self.assertFalse(report["record_by_record_official_provenance_verified"])

    def test_gap_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=fixture(folder,"539")
            text=path.read_text().replace("115000003","115000007",1)
            path.write_text(text)
            with self.assertRaisesRegex(ValueError,"nonconsecutive"):
                audit_csv(path,"539")

    def test_invalid_special_number(self):
        with tempfile.TemporaryDirectory() as folder:
            path=fixture(folder,"lotto")
            path.write_text(path.read_text().replace(",07\n",",06\n",1))
            with self.assertRaisesRegex(ValueError,"special"):
                audit_csv(path,"lotto")

    def test_date_and_year_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=fixture(folder,"539")
            path.write_text(path.read_text().replace("2026-01-01","2025-01-01",1))
            with self.assertRaisesRegex(ValueError,"ROC"):
                audit_csv(path,"539")

    def test_out_of_order_date(self):
        with tempfile.TemporaryDirectory() as folder:
            path=fixture(folder,"539")
            path.write_text(path.read_text().replace("2026-01-03","2026-01-01",1))
            with self.assertRaisesRegex(ValueError,"nonincreasing"):
                audit_csv(path,"539")


    def test_official_comparison_rejects_missing_or_changed_values(self):
        from of_lgbm_real_history_gate import compare_official_normalized
        for game in ("lotto", "539"):
            with tempfile.TemporaryDirectory() as folder:
                snapshot = fixture(folder, game)
                official = Path(folder) / "official_normalized.csv"
                official.write_bytes(snapshot.read_bytes())
                passed = compare_official_normalized(snapshot, official, game)
                self.assertTrue(passed["official_rows_verified"])
                self.assertEqual(passed["matched_draw_ids"], 20)
                self.assertFalse(passed.get("independent_official_provenance_verified", False))
                # Equal normalized CSV bytes are not independent official authentication.
                changed = official.read_text().replace("2026-01-03", "2026-01-04", 1)
                official.write_text(changed)
                result = compare_official_normalized(snapshot, official, game)
                self.assertFalse(result["official_rows_verified"])
                self.assertIn("date_mismatch", {r["reason"] for r in result["issues"]})
                official.write_bytes(snapshot.read_bytes())
                changed = official.read_text().replace("01 02 03 04 05", "01 02 03 04 08", 1)
                official.write_text(changed)
                result = compare_official_normalized(snapshot, official, game)
                self.assertFalse(result["official_rows_verified"])
                self.assertIn("first_zone_mismatch", {r["reason"] for r in result["issues"]})

    def test_official_comparison_rejects_missing_and_duplicate_ids(self):
        from of_lgbm_real_history_gate import compare_official_normalized
        with tempfile.TemporaryDirectory() as folder:
            snapshot = fixture(folder, "539")
            official = Path(folder) / "official_normalized.csv"
            lines = snapshot.read_text().splitlines()
            official.write_text("\\n".join(lines[:-1]) + "\\n")
            result = compare_official_normalized(snapshot, official, "539")
            self.assertFalse(result["official_rows_verified"])
            self.assertIn("official_draw_missing", {r["reason"] for r in result["issues"]})
            official.write_text("\\n".join(lines + [lines[1]]) + "\\n")
            with self.assertRaisesRegex(ValueError, "duplicate official draw_id"):
                compare_official_normalized(snapshot, official, "539")

    def test_official_comparison_rejects_power_snapshot_without_dates(self):
        from of_lgbm_real_history_gate import compare_official_normalized
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "power.csv"
            official = Path(folder) / "power_normalized.csv"
            with archive.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["draw_id", "first", "second"])
                writer.writerow(["115000001", "01 02 03 04 05 06", "2"])
            with official.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["draw_id", "draw_date", "first", "second"])
                writer.writerow(["115000001", "2026-01-01", "01 02 03 04 05 06", "2"])
            result = compare_official_normalized(archive, official, "power")
            self.assertFalse(result["official_rows_verified"])
            self.assertIn("snapshot_date_missing", {r["reason"] for r in result["issues"]})

if __name__=="__main__":unittest.main()
