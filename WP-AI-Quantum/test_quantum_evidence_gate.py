import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from quantum_evidence_gate import gate
class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.selection={"status":"receipt_contract_verified","real_qpu_verified":False,"qpu_influenced_selection":False,
                        "qpu_job_id":"MOCK-123","backend":"ibm_mock","selection_valid":True}
        self.provider={"provider_provenance_independently_verified":True,"job_id":"MOCK-123",
                       "backend":"ibm_mock","quantum_lottery_optimization_executed":False}
    def test_matching_contract(self):
        self.assertTrue(gate(self.selection,self.provider)["hardware_probe_verified"])
    def test_mismatched_job_rejected(self):
        self.provider["job_id"]="OTHER"
        with self.assertRaises(ValueError):gate(self.selection,self.provider)
    def test_unverified_provider_rejected(self):
        self.provider["provider_provenance_independently_verified"]=False
        with self.assertRaises(ValueError):gate(self.selection,self.provider)
    def test_reject_invalid_selection(self):
        self.selection["selection_valid"]=False
        with self.assertRaises(ValueError):gate(self.selection,self.provider)
    def test_not_a_lottery_qpu_prediction(self):
        self.provider["quantum_lottery_optimization_executed"]=True
        with self.assertRaises(ValueError):gate(self.selection,self.provider)
if __name__=="__main__":unittest.main()
