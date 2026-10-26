import csv
import tempfile
import unittest
from pathlib import Path
from evidence_audit import audit_history, walkforward, verify_forecast_ledger
from wp_os_git_max import load

ROOT=Path(__file__).parent

class EvidenceAuditTests(unittest.TestCase):
    def test_actual_archive_audit_and_walkforward(self):
        p=ROOT/"historical_archive.csv"
        meta=audit_history(p)
        self.assertGreaterEqual(meta["count"],25)
        d=load(p)
        out=walkforward(d,100)
        self.assertEqual(out["tested"],len(d)-10)
        self.assertEqual(sum(out["first_zone_hit_distribution"].values()),out["tested"])
        self.assertEqual(len(out["observations"]),out["tested"])
        self.assertEqual(out["observations"][0]["through_draw"],d[9]["draw_id"])
        self.assertEqual(out["observations"][0]["target_draw"],d[10]["draw_id"])

    def test_invalid_lengths(self):
        d=load(ROOT/"historical_archive.csv")
        with self.assertRaises(ValueError): walkforward(d[:10])
        with self.assertRaises(ValueError): walkforward(d,0)

    def test_ledger_checks(self):
        d=load(ROOT/"historical_archive.csv")
        self.assertEqual(verify_forecast_ledger(None,d)["status"],"not_supplied")
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"forecasts.csv"
            def write(rows):
                with p.open("w",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f,fieldnames=["target_draw","through_draw","first","second","prediction_sha256"])
                    w.writeheader();w.writerows(rows)
            good={"target_draw":d[-1]["draw_id"],"through_draw":d[-2]["draw_id"],
                  "first":"01 02 03 04 05 06","second":"03","prediction_sha256":"dummy"}
            write([good])
            r=verify_forecast_ledger(str(p),d)
            self.assertEqual(r["matched"],1)
            write([good,good])
            with self.assertRaisesRegex(ValueError,"Duplicate"):verify_forecast_ledger(str(p),d)
            write([{**good,"first":"01 01 02 03 04 05"}])
            with self.assertRaisesRegex(ValueError,"Invalid forecast"):verify_forecast_ledger(str(p),d)
            write([{**good,"through_draw":good["target_draw"]}])
            with self.assertRaisesRegex(ValueError,"ordering"):verify_forecast_ledger(str(p),d)
            write([{**good,"prediction_sha256":""}])
            with self.assertRaisesRegex(ValueError,"Missing"):verify_forecast_ledger(str(p),d)
if __name__=="__main__":unittest.main()
