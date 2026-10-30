import csv,json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from quantum_e2e_validation import validate
class TestE2E(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        base=pathlib.Path(self.tmp.name)
        self.csv=base/"draws.csv";self.receipt=base/"qpu.json"
        with self.csv.open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["draw_id","first","second"])
            for i in range(20):w.writerow([str(i+1),"01 02 03 04 05 06",str(i%8+1)])
    def test_selection_independent(self):
        r=validate(self.csv)
        self.assertTrue(r["selection_valid"])
        self.assertFalse(r["real_qpu_verified"])
        self.assertFalse(r["qpu_influenced_selection"])
    def test_reject_unfinished_receipt(self):
        self.receipt.write_text(json.dumps({"status":"submitted","job_id":"fake"}))
        with self.assertRaises(ValueError):validate(self.csv,self.receipt)
    def test_receipt_contract_is_not_provider_proof(self):
        self.receipt.write_text(json.dumps({"status":"completed","real_qpu_result_verified":True,
            "job_id":"SYNTHETIC-TEST","backend":"test","shots":8,"counts":{"00":4,"11":4}}))
        r=validate(self.csv,self.receipt)
        self.assertFalse(r["provider_provenance_independently_verified"])
        self.assertFalse(r["qpu_influenced_selection"])
if __name__=="__main__":unittest.main()
