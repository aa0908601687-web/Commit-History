import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from workflow_safety_contract import validate_workflows
FREE="""on:
  workflow_dispatch:
jobs:
  preflight:
    env:
      IBM_QUANTUM_API_TOKEN: ''
      IBM_QUANTUM_INSTANCE: ''
"""
HARDWARE="""on:
  workflow_dispatch:
    inputs:
      authorize_qpu_cost:
        default: false
jobs:
  run:
    if: ${{ inputs.authorize_qpu_cost == true }}
    steps:
      - run: python ibm_provider_verify.py
      - run: python quantum_evidence_gate.py
"""
class Tests(unittest.TestCase):
    def test_safe(self):self.assertTrue(validate_workflows(FREE,HARDWARE))
    def test_reject_hardware(self):
        with self.assertRaises(ValueError):validate_workflows(FREE+"    run: python ibm_real_qpu_probe.py",HARDWARE)
    def test_reject_unprotected_hardware(self):
        with self.assertRaises(ValueError):validate_workflows(FREE,HARDWARE.replace("if: ${{ inputs.authorize_qpu_cost == true }}","if: true"))
if __name__=="__main__":unittest.main()
