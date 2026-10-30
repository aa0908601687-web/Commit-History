#!/usr/bin/env python3
"""Opt-in IBM real-QPU probe. Never label a simulator or missing result as QPU success."""
import argparse, json, os
from pathlib import Path

def save(path,record):
    Path(path).write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def run_probe(output,service=None,backend_name=None,shots=128):
    if not 1 <= shots <= 4096: raise ValueError("shots must be 1..4096")
    if service is None:
        token=os.environ.get("IBM_QUANTUM_API_TOKEN")
        instance=os.environ.get("IBM_QUANTUM_INSTANCE")
        if not token or not instance:
            raise RuntimeError("IBM credentials missing; no hardware submission")
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    if service is None:
        service=QiskitRuntimeService(channel="ibm_quantum_platform",token=token,instance=instance)
    backend=service.backend(backend_name) if backend_name else service.least_busy(operational=True,simulator=False,min_num_qubits=2)
    configuration = backend.configuration() if callable(getattr(backend,"configuration",None)) else None
    if getattr(backend,"simulator",False) or getattr(configuration,"simulator",False):
        raise RuntimeError("Simulator rejected")
    if getattr(backend,"num_qubits",0)<2: raise RuntimeError("At least 2 physical qubits required")
    qc=QuantumCircuit(2,2)
    qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
    isa=transpile(qc,backend=backend,optimization_level=1)
    job=SamplerV2(mode=backend).run([isa],shots=shots)
    record={"backend":str(backend.name),"job_id":str(job.job_id()),"shots":shots,
            "status":"submitted","real_qpu_result_verified":False,
            "quantum_lottery_optimization_executed":False,"counts":None}
    save(output,record)
    print("SUBMITTED "+json.dumps(record),flush=True)
    try:
        counts=job.result()[0].data.c.get_counts()
        if sum(counts.values()) != shots:
            raise RuntimeError("Returned counts do not match shots")
        record.update(status="completed",real_qpu_result_verified=True,counts=counts)
        save(output,record)
    except Exception as exc:
        record.update(status="result_failed",error_type=type(exc).__name__)
        save(output,record)
        raise
    return record

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backend")
    p.add_argument("--shots",type=int,default=128)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(run_probe(args.output,backend_name=args.backend,shots=args.shots),indent=2))
