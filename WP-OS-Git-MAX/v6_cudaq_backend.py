"""Optional NVIDIA CUDA-Q *diagnostic* local circuit; selection is exact Python.

No QAOA, remote QPU, or variational optimization is performed here.
The qpp-cpu circuit checks runtime integration only and cannot alter the ticket.
"""
from __future__ import annotations
import importlib
from typing import Mapping, Sequence

TARGET="qpp-cpu"
SEED=20261103

def availability() -> dict:
    try:
        cudaq=importlib.import_module("cudaq")
    except (ImportError, OSError) as exc:
        return {"installed":False,"backend":TARGET,"qpu":False,
                "reason":f"{type(exc).__name__}: {exc}"}
    return {"installed":True,"backend":TARGET,"qpu":False,
            "version":getattr(cudaq,"__version__","unknown")}

def _normalise(pool: Sequence[int], scores: Mapping[int,float]) -> list[float]:
    values=[float(scores[n]) for n in pool]
    low,high=min(values),max(values)
    if high==low:return [0.5]*len(values)
    return [(v-low)/(high-low) for v in values]

def select_six(pool: Sequence[int], scores: Mapping[int,float], *, required:bool=False):
    if len(pool)!=8 or len(set(pool))!=8:
        raise ValueError("CUDA-Q diagnostic requires eight distinct candidates")
    if any(n not in scores for n in pool):
        raise ValueError("missing CUDA-Q candidate score")
    from v6_exact_search import exact_select
    selection,_,_=exact_select(pool,scores)
    status=availability()
    if not status["installed"]:
        if required:raise RuntimeError("CUDA-Q is required but unavailable: "+status["reason"])
        return None,{**status,"enabled":True,"executed":False,"fallback":"Python exact enumeration"}
    cudaq=importlib.import_module("cudaq")
    cudaq.set_target(TARGET)
    if hasattr(cudaq,"set_random_seed"):cudaq.set_random_seed(SEED)
    # A one-qubit smoke test of the local runtime, not a combinatorial optimizer.
    kernel=cudaq.make_kernel()
    qubit=kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)
    counts=cudaq.sample(kernel,shots_count=16)
    if sum(int(v) for v in counts.values())!=16:
        raise RuntimeError("CUDA-Q local diagnostic returned an invalid shot count")
    return selection,{**status,"enabled":True,"executed":True,"seed":SEED,
                      "diagnostic_only":True,"selection_backend":"python-exact",
                      "real_qpu_connected":False,"remote_service_connected":False,
                      "predictive_advantage_verified":False}
