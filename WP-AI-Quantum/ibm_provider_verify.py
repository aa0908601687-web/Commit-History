#!/usr/bin/env python3
"""Cross-check a saved QPU receipt against IBM Quantum job API. Never submit a new job."""
import argparse,json,os
from pathlib import Path

def verify(receipt,service):
    if receipt.get("status")!="completed" or not receipt.get("job_id"):
        raise ValueError("Receipt does not describe a completed job")
    job=service.job(receipt["job_id"])
    status=str(job.status()).strip().upper().split(".")[-1]
    if status not in {"DONE", "COMPLETED"}:
        raise RuntimeError("IBM job is not complete: "+status)
    backend=job.backend()
    name=str(backend.name)
    configuration=backend.configuration() if callable(getattr(backend,"configuration",None)) else None
    if getattr(backend,"simulator",False) or getattr(configuration,"simulator",False):
        raise RuntimeError("IBM job was not on a real QPU")
    if name!=receipt["backend"]: raise ValueError("Backend mismatch")
    counts=job.result()[0].data.c.get_counts()
    shots=receipt.get("shots")
    if type(shots) is not int or not 1<=shots<=4096:
        raise ValueError("Invalid requested shot count")
    if (not isinstance(counts,dict) or not counts or
        any(k not in {"00","01","10","11"} or type(v) is not int or v<0 for k,v in counts.items()) or
        counts!=receipt["counts"] or sum(counts.values())!=shots):
        raise ValueError("Provider measurement mismatch")
    return {"provider_provenance_independently_verified":True,
            "job_id":receipt["job_id"],"backend":name,
            "shots":receipt["shots"],"quantum_lottery_optimization_executed":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    token=os.environ.get("IBM_QUANTUM_API_TOKEN")
    instance=os.environ.get("IBM_QUANTUM_INSTANCE")
    if not token or not instance: raise RuntimeError("Provider credentials missing")
    from qiskit_ibm_runtime import QiskitRuntimeService
    service=QiskitRuntimeService(channel="ibm_quantum_platform",token=token,instance=instance)
    report=verify(json.loads(Path(args.receipt).read_text()),service)
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":main()
