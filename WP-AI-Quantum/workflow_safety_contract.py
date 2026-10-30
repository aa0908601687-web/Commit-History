#!/usr/bin/env python3
"""Static CI contract: offline preflight must not submit IBM Quantum hardware jobs."""
from pathlib import Path
import argparse

def validate_workflows(free_text,hardware_text):
    free_job=free_text.split("jobs:",1)[-1]
    if "workflow_dispatch:" not in free_text:
        raise ValueError("Missing manual offline trigger")
    if "IBM_QUANTUM_API_TOKEN: ''" not in free_job or "IBM_QUANTUM_INSTANCE: ''" not in free_job:
        raise ValueError("Offline preflight must blank IBM credentials")
    forbidden=("ibm_real_qpu_probe.py","SamplerV2(","QiskitRuntimeService(",
               "IBM_QUANTUM_API_TOKEN: ${{ secrets.","authorize_qpu_cost: true")
    if any(value in free_job for value in forbidden):
        raise ValueError("Offline preflight contains hardware or credential submission reference")
    if "workflow_dispatch:" not in hardware_text or "default: false" not in hardware_text:
        raise ValueError("Hardware action must require manual opt in")
    if "if: ${{ inputs.authorize_qpu_cost == true }}" not in hardware_text:
        raise ValueError("Hardware job must be opt-in")
    if "ibm_provider_verify.py" not in hardware_text or "quantum_evidence_gate.py" not in hardware_text:
        raise ValueError("Hardware provider validation gate missing")
    return True

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--free",required=True);p.add_argument("--hardware",required=True)
    a=p.parse_args()
    validate_workflows(Path(a.free).read_text(),Path(a.hardware).read_text())
    print("Offline/hardware separation contract passed (static only; no QPU executed)")
