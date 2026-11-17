import json
import tempfile
import unittest
from pathlib import Path
from service import register_report, list_runs, send_alert

class ServiceTests(unittest.TestCase):
    def test_register_and_list(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            report=p/"report.json"
            report.write_text(json.dumps({"success":True}))
            registered=register_report(p/"runs.db",report,model="kumo")
            self.assertEqual(registered["id"],1)
            self.assertEqual(list_runs(p/"runs.db")[0]["model"],"kumo")
            self.assertEqual(len(registered["sha256"]),64)
    def test_invalid_report_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            f=p/"bad.json"
            f.write_text("not json")
            with self.assertRaises(json.JSONDecodeError):
                register_report(p/"runs.db",f)
    def test_no_webhook(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"KUMO_ALERT_WEBHOOK":""}):
            self.assertFalse(send_alert("test")["sent"])
if __name__=="__main__":
    unittest.main()
