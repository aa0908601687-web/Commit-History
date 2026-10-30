import json, pathlib, sys, tempfile, unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from ibm_real_qpu_probe import save, run_probe
class QpuContractTests(unittest.TestCase):
    def test_receipt_persists(self):
        with tempfile.TemporaryDirectory() as d:
            target=pathlib.Path(d)/"receipt.json"
            save(target,{"job_id":"test-only","status":"submitted"})
            self.assertEqual(json.loads(target.read_text())["status"],"submitted")
    def test_missing_credentials_rejected_before_sdk_or_network(self):
        with patch.dict("os.environ",{"IBM_QUANTUM_API_TOKEN":"","IBM_QUANTUM_INSTANCE":""}):
            with self.assertRaisesRegex(RuntimeError,"credentials missing"):
                run_probe("unused.json")
    def test_invalid_shots_rejected_before_sdk_or_network(self):
        with self.assertRaises(ValueError):
            run_probe("unused.json",shots=0)
        with self.assertRaises(ValueError):
            run_probe("unused.json",shots=4097)
if __name__=="__main__":unittest.main()
