#!/usr/bin/env python3
"""Final evidence gate. Reject mixed job records and synthetic-only proof."""
import argparse,json
from pathlib import Path

def gate(selection,provider):
    if selection.get("selection_valid") is not True:
        raise ValueError("Independent lottery selection invalid")
    if selection.get("status")!="receipt_contract_verified":
        raise ValueError("Receipt contract not validated")
    if selection.get("real_qpu_verified") is not False or selection.get("qpu_influenced_selection") is not False:
        raise ValueError("Selection report provenance is inconsistent")
    if provider.get("provider_provenance_independently_verified") is not True:
        raise ValueError("Provider evidence missing")
    if not provider.get("job_id") or provider["job_id"] != selection.get("qpu_job_id"):
        raise ValueError("QPU job ID mismatch")
    if provider.get("backend") != selection.get("backend"):
        raise ValueError("QPU backend mismatch")
    if provider.get("quantum_lottery_optimization_executed") is not False:
        raise ValueError("Hardware probe is not lottery optimization")
    return {"verification":"provider_job_cross_checked",
            "hardware_probe_verified":True,
            "qpu_job_id":provider["job_id"],
            "backend":provider["backend"],
            "selection_valid":selection["selection_valid"],
            "qpu_influenced_selection":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selection",required=True)
    p.add_argument("--provider",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=gate(json.loads(Path(a.selection).read_text()),json.loads(Path(a.provider).read_text()))
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
