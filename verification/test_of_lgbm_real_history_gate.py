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

if __name__=="__main__":unittest.main()
