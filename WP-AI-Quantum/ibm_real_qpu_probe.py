#!/usr/bin/env python3
"""IBM real-QPU probe. Fail closed: never substitute a simulator for hardware."""
import argparse, json, os

def run_probe(service=None, backend_name=None, shots=128):
    if not 1 <= shots <= 4096: raise ValueError("shots outside conservative probe range")
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    if service is None:
        token=os.environ.get("IBM_QUANTUM_API_TOKEN")
        instance=os.environ.get("IBM_QUANTUM_INSTANCE")
        if not token or not instance:
            raise RuntimeError("IBM_QUANTUM_API_TOKEN and IBM_QUANTUM_INSTANCE required; no hardware job submitted")
        service=QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    backend=service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False, min_num_qubits=2)
    if getattr(backend.configuration(),"simulator",False) or getattr(backend,"simulator",False):
        raise RuntimeError("Simulation backend rejected: real QPU required")
    if getattr(backend,"num_qubits",0)<2: raise RuntimeError("Backend lacks required qubits")
    circuit=QuantumCircuit(2,2)
    circuit.h(0); circuit.cx(0,1); circuit.measure([0,1],[0,1])
    isa=transpile(circuit,backend=backend,optimization_level=1)
    sampler=SamplerV2(mode=backend)
    job=sampler.run([isa],shots=shots)
    result={"mode":"real_qpu_requested","backend":backend.name,"job_id":job.job_id(),"shots":shots,
            "status":"submitted","counts":None,"lottery_prediction":False}
    print(json.dumps(result),flush=True)
    raw=job.result()[0]
    # Qiskit register name for QuantumCircuit(2,2) is 'c'.
    result["counts"]=raw.data.c.get_counts()
    result["status"]="completed"
    result["mode"]="real_qpu_completed"
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backend",default=None)
    p.add_argument("--shots",type=int,default=128)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=run_probe(backend_name=args.backend,shots=args.shots)
    with open(args.output,"w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
