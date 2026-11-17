import tempfile
import unittest
from pathlib import Path
import pandas as pd
from operations import drift, fingerprint, ingest, manifest

class OperationsTests(unittest.TestCase):
    def test_local_ingest_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            src, dest, out = (Path(t)/x for x in ("src.csv","dst.csv","manifest.json"))
            src.write_text("time,target,x\n1,0,1\n2,1,2\n")
            p, df = ingest(str(src), str(dest))
            self.assertEqual(len(df), 2)
            result = manifest([p], out)
            self.assertEqual(result["files"][str(dest)]["sha256"], fingerprint(dest))
    def test_http_rejected(self):
        with self.assertRaises(ValueError):
            ingest("http://example.com/data.csv", "unused.csv")
    def test_drift(self):
        a = pd.DataFrame({"x": [1,2,3,4,5]})
        b = pd.DataFrame({"x": [11,12,13,14,15]})
        self.assertTrue(drift(a,b,["x"])["x"]["alert"])
if __name__ == "__main__":
    unittest.main()
