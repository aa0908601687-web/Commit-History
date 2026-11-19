"""No fabricated dates, no number changes, no partial Power date merges."""
import csv
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_power_date_reconcile import reconcile

class PowerDatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p=Path(self.tmp.name)
        self.old=p/"old.csv";self.source=p/"official_normalized.csv"
        self.output=p/"dated.csv";self.report=p/"audit.json"
        self.write(False)

    def write(self, wrong=False, missing=False, dates=True):
        with self.old.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["draw_id","first","second"])
            for n in range(1,5): w.writerow([f"115{n:06d}","01 02 03 04 05 06",2])
        with self.source.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["draw_id","draw_date","first","second"])
            for n in range(1, 4 if missing else 5):
                day=(date(2026,1,1)+timedelta(days=n-1)).isoformat()
                w.writerow([f"115{n:06d}",day,"01 02 03 04 05 07" if wrong and n==3 else "01 02 03 04 05 06",2])

    def test_exact_match_produces_dated_new_file(self):
        result=reconcile(self.old,self.source,self.output,self.report)
        self.assertEqual(result["dated_rows"],4)
        self.assertTrue(result["numbers_preserved"])
        self.assertFalse(result["independent_official_provenance_verified"])
        with self.output.open(newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(rows[-1]["draw_date"],"2026-01-04")
        self.assertEqual(rows[-1]["first"],"01 02 03 04 05 06")
        self.assertEqual(self.old.read_text().splitlines()[0],"draw_id,first,second")

    def test_rejects_wrong_number_without_output(self):
        self.write(wrong=True)
        with self.assertRaisesRegex(ValueError,"numbers differ"):
            reconcile(self.old,self.source,self.output,self.report)
        self.assertFalse(self.output.exists())

    def test_rejects_missing_date_row_without_output(self):
        self.write(missing=True)
        with self.assertRaisesRegex(ValueError,"source draw IDs differ"):
            reconcile(self.old,self.source,self.output,self.report)
        self.assertFalse(self.output.exists())

    def test_report_creation_failure_removes_partial_output(self):
        # A bad report destination must not leave a misleading dated archive.
        self.report = Path(self.tmp.name) / "missing_directory" / "audit.json"
        with self.assertRaises(FileNotFoundError):
            reconcile(self.old, self.source, self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())
        self.assertTrue(self.old.exists())

    def test_never_overwrites_original(self):
        with self.assertRaisesRegex(ValueError,"must not overwrite"):
            reconcile(self.old,self.source,self.old,self.report)

    def test_rejects_swapped_draw_identifiers(self):
        # Both zones are intentionally identical in the fixture: ID order check
        # must catch a mislabeled official date even when numbers appear equal.
        data=self.source.read_text()
        data=data.replace("115000002", "TEMP_DRAW").replace("115000003", "115000002").replace("TEMP_DRAW", "115000003")
        self.source.write_text(data)
        with self.assertRaisesRegex(ValueError,"nonincreasing draw ID"):
            reconcile(self.old,self.source,self.output,self.report)
        self.assertFalse(self.output.exists())

    def test_rejects_reversed_archive_order(self):
        rows=self.old.read_text().splitlines()
        rows[2],rows[3]=rows[3],rows[2]
        self.old.write_text(chr(10).join(rows)+chr(10))
        with self.assertRaisesRegex(ValueError,"nonincreasing draw ID"):
            reconcile(self.old,self.source,self.output,self.report)
        self.assertFalse(self.output.exists())

    def test_rejects_roc_year_mismatch(self):
        data=self.source.read_text().replace("2026-01-01","2025-01-01")
        self.source.write_text(data)
        with self.assertRaisesRegex(ValueError,"ISO date/ROC"):
            reconcile(self.old,self.source,self.output,self.report)

if __name__=="__main__":
    unittest.main()
