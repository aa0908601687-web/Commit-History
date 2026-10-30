import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from ibm_provider_verify import verify
class Backend:
    name="ibm_test_qpu"
    simulator=False
    def configuration(self):return self
class Counts:
    def get_counts(self):return {"00":5,"11":3}
class Data:c=Counts()
class Result:data=Data()
class Job:
    def status(self):return "DONE"
    def backend(self):return Backend()
    def result(self):return [Result()]
class Service:
    def job(self,job_id):return Job()
class TestProvider(unittest.TestCase):
    def setUp(self):
        self.receipt={"status":"completed","job_id":"SYNTHETIC","backend":"ibm_test_qpu","shots":8,"counts":{"00":5,"11":3}}
    def test_provider_contract(self):
        self.assertTrue(verify(self.receipt,Service())["provider_provenance_independently_verified"])
    def test_reject_mismatch(self):
        self.receipt["counts"]={"00":8}
        with self.assertRaises(ValueError):verify(self.receipt,Service())
    def test_reject_misleading_status(self):
        class MisleadingJob(Job):
            def status(self): return "NOT_DONE"
        class MisleadingService(Service):
            def job(self,job_id): return MisleadingJob()
        with self.assertRaises(RuntimeError):verify(self.receipt,MisleadingService())
    def test_reject_invalid_shots(self):
        self.receipt["shots"]=True
        with self.assertRaises(ValueError):verify(self.receipt,Service())
    def test_reject_incomplete(self):
        self.receipt["status"]="submitted"
        with self.assertRaises(ValueError):verify(self.receipt,Service())
if __name__=="__main__":unittest.main()
