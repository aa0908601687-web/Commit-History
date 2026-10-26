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
        self.assertEqual(walkforward(d)["tested"],len(d)-10)
        expected=min(100,len(d)-10)
        self.assertEqual(out["tested"],expected)
        self.assertEqual(sum(out["first_zone_hit_distribution"].values()),out["tested"])
        self.assertEqual(len(out["observations"]),out["tested"])
        start=len(d)-expected
        self.assertEqual(out["observations"][0]["through_draw"],d[start-1]["draw_id"])
        self.assertEqual(out["observations"][0]["target_draw"],d[start]["draw_id"])

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

    def test_history_errors_and_empty_ledger(self):
        from evidence_audit import audit_history
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"history.csv"
            p.write_text("draw_id,first,second\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Empty"): audit_history(p)
            d=load(ROOT/"historical_archive.csv")
            r1=d[0]
            r2=d[1]
            p.write_text("draw_id,first,second\n"
                +f"{r2['draw_id']},{' '.join(map(str,r2['first']))},{r2['second']}\n"
                +f"{r1['draw_id']},{' '.join(map(str,r1['first']))},{r1['second']}\n",encoding="utf-8")
            with self.assertRaises(ValueError): audit_history(p)
            ledger=Path(td)/"forecast.csv"
            ledger.write_text("target_draw,through_draw,first,second,prediction_sha256\n",encoding="utf-8")
            self.assertEqual(verify_forecast_ledger(ledger,d)["matched"],0)
            ledger.write_text("other,bad\nfoo,bar\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"missing"): verify_forecast_ledger(ledger,d)

    def test_cli_and_ledger_unmatched(self):
        import contextlib
        import io
        import json
        import sys
        from unittest.mock import patch
        import evidence_audit
        d=load(ROOT/"historical_archive.csv")
        with tempfile.TemporaryDirectory() as td:
            ledger=Path(td)/"ledger.csv"
            with ledger.open("w",newline="",encoding="utf-8") as fp:
                w=csv.DictWriter(fp,fieldnames=["target_draw","through_draw","first","second","prediction_sha256"])
                w.writeheader()
                w.writerow({"target_draw":"999999999","through_draw":d[-1]["draw_id"],"first":"01 02 03 04 05 06","second":"1","prediction_sha256":"abc"})
            self.assertEqual(verify_forecast_ledger(ledger,d)["matched"],0)
            out=Path(td)/"report.json"
            with patch.object(sys,"argv",["evidence_audit.py","--csv",str(ROOT/"historical_archive.csv"),"--backtest-count","100","--ledger",str(ledger),"--output",str(out)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    evidence_audit.main()
            report=json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["walkforward"]["tested"],min(100,len(d)-10))
            self.assertEqual(report["validation"]["full_requested_backtest"],len(d)-10>=100)
            with patch.object(sys,"argv",["evidence_audit.py","--csv",str(ROOT/"historical_archive.csv"),"--backtest-count","5"]):
                with contextlib.redirect_stdout(io.StringIO()) as capture:
                    evidence_audit.main()
            self.assertEqual(json.loads(capture.getvalue())["walkforward"]["tested"],5)
            short=Path(td)/"short.csv"
            short.write_text((ROOT/"historical_archive.csv").read_text(encoding="utf-8").splitlines()[0]+"\n"
                +"\n".join((ROOT/"historical_archive.csv").read_text(encoding="utf-8").splitlines()[1:6])+"\n",encoding="utf-8")
            with patch.object(sys,"argv",["evidence_audit.py","--csv",str(short)]):
                with contextlib.redirect_stdout(io.StringIO()) as capture:
                    evidence_audit.main()
            self.assertEqual(json.loads(capture.getvalue())["walkforward"]["status"],"insufficient_history")

    def test_script_entrypoint_and_non_numeric_target(self):
        import subprocess
        import sys
        import json
        d=load(ROOT/"historical_archive.csv")
        with tempfile.TemporaryDirectory() as td:
            ledger=Path(td)/"ledger.csv"
            with ledger.open("w",newline="",encoding="utf-8") as fp:
                w=csv.DictWriter(fp,fieldnames=["target_draw","through_draw","first","second","prediction_sha256"])
                w.writeheader()
                w.writerow({"target_draw":"not-a-draw","through_draw":d[-1]["draw_id"],
                            "first":"01 02 03 04 05 06","second":"03","prediction_sha256":"hash"})
            with self.assertRaisesRegex(ValueError,"ordering"):
                verify_forecast_ledger(ledger,d)
            result=subprocess.run([sys.executable,str(ROOT/"evidence_audit.py"),
                "--csv",str(ROOT/"historical_archive.csv"),"--backtest-count","100"],
                capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)["history"]["count"],len(d))
    def test_monotonic_audit_guard_and_in_process_entrypoint(self):
        import contextlib
        import io
        import runpy
        import sys
        from unittest.mock import patch
        import evidence_audit
        d=load(ROOT/"historical_archive.csv")
        with patch.object(evidence_audit,"load",return_value=[d[1],d[0]]):
            with self.assertRaisesRegex(ValueError,"not increasing"):
                audit_history(ROOT/"historical_archive.csv")
        with patch.object(sys,"argv",["evidence_audit.py","--csv",str(ROOT/"historical_archive.csv"),"--backtest-count","100"]):
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                runpy.run_path(str(ROOT/"evidence_audit.py"),run_name="__main__")
        self.assertIn(f'"tested": {min(100,len(d)-10)}',captured.getvalue())
if __name__=="__main__":unittest.main()
