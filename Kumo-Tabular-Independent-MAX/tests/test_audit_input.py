"""Unit and lightweight integration test for CSV audit CLI."""
import json
import tempfile
import unittest
from pathlib import Path
from audit_input import main

class AuditInputTests(unittest.TestCase):
    def test_manifest_is_generated_and_has_split(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            csvfile = folder / "input.csv"
            csvfile.write_text("date,feature,target\n2026-01-01,1,A\n2026-01-02,2,B\n2026-01-03,3,A\n2026-01-04,4,B\n2026-01-05,5,A\n", encoding="utf-8")
            outfile = folder / "result.json"
            main(["--csv", str(csvfile), "--target", "target", "--time-column", "date", "--output", str(outfile)])
            result = json.loads(outfile.read_text(encoding="utf-8"))
            self.assertEqual(result["metadata"]["train_rows"], 4)
            self.assertEqual(result["metadata"]["test_rows"], 1)
            self.assertEqual(len(result["source_sha256"]), 64)
    def test_missing_target_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"input.csv"
            p.write_text("x,y\na,1\nb,2\nc,3\n")
            with self.assertRaises(ValueError):
                main(["--csv",str(p),"--target","bad"])
if __name__ == "__main__":
    unittest.main()
