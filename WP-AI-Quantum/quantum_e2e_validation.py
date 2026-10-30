#!/usr/bin/env python3
"""Validate independent selection and authentic QPU receipt without claiming QPU selection."""
import argparse,json,pathlib,sys
from wp_quantum_max import load,select

def validate(csv_path,receipt_path=None):
    prediction=select(load(csv_path))
    assert prediction["abe_used"] is False and prediction["quantum_hardware_used"] is False
    assert len(prediction["first"])==len(set(prediction["first"]))==6
    assert all(1<=x<=38 for x in prediction["first"]) and 1<=prediction["second"]<=8
    summary={"selection_valid":True,"through_draw":prediction["through_draw"],
             "selection_sha256":prediction["sha256"],"real_qpu_verified":False,
             "qpu_influenced_selection":False,"status":"selection_only"}
    if receipt_path:
        receipt=json.loads(pathlib.Path(receipt_path).read_text())
        if (receipt.get("status")!="completed" or receipt.get("real_qpu_result_verified") is not True
            or not receipt.get("job_id") or not receipt.get("backend")
            or not isinstance(receipt.get("counts"),dict)
            or sum(receipt["counts"].values())!=receipt.get("shots")
            or not all(k in {"00","01","10","11"} and isinstance(v,int) and v>=0 for k,v in receipt["counts"].items())):
            raise ValueError("Incomplete or invalid QPU receipt; real hardware not verified")
        summary.update(real_qpu_verified=True,status="receipt_contract_verified",
                       qpu_job_id=receipt["job_id"],backend=receipt["backend"])
        # The local receipt is not independent cryptographic proof of hardware execution.
        summary["provider_provenance_independently_verified"]=False
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--receipt")
    args=p.parse_args()
    print(json.dumps(validate(args.csv,args.receipt),indent=2))
