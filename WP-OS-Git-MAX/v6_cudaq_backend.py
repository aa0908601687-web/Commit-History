"""Optional NVIDIA CUDA-Q qpp-cpu capability verification for V6.

The backend runs a non-variational single-qubit simulator probe and delegates
candidate selection to the exhaustive 8-choose-6 Python optimizer. No QAOA,
parameterized mixer, QPU or remote execution.
"""
from __future__ import annotations
import importlib
from typing import Mapping, Sequence

TARGET = "qpp-cpu"
SEED = 20261103

def availability() -> dict:
    try:
        cudaq = importlib.import_module("cudaq")
    except (ImportError, OSError) as exc:
        return {"installed": False, "backend": TARGET, "qpu": False,
                "reason": f"{type(exc).__name__}: {exc}"}
    return {"installed": True, "backend": TARGET, "qpu": False,
            "version": getattr(cudaq, "__version__", "unknown")}

def _normalise(pool: Sequence[int], scores: Mapping[int, float]) -> list[float]:
    values = [float(scores[n]) for n in pool]
    low, high = min(values), max(values)
    return [0.5] * len(values) if high == low else [(v-low)/(high-low) for v in values]

def select_six(pool: Sequence[int], scores: Mapping[int, float], *,
               required: bool=False) -> tuple[list[int] | None, dict]:
    """Verify CUDA-Q CPU execution and return the exact optimum of 28 candidates."""
    if len(pool) != 8 or len(set(pool)) != 8:
        raise ValueError("CUDA-Q backend requires eight distinct candidates")
    if any(n not in scores for n in pool):
        raise ValueError("missing CUDA-Q candidate score")
    status = availability()
    if not status["installed"]:
        if required:
            raise RuntimeError("CUDA-Q is required but unavailable: " + status["reason"])
        return None, {**status, "enabled": True, "executed": False,
                      "fallback": "Python exact 8-choose-6 enumeration"}

    cudaq = importlib.import_module("cudaq")
    cudaq.set_target(TARGET)
    if hasattr(cudaq, "set_random_seed"):
        cudaq.set_random_seed(SEED)

    @cudaq.kernel
    def simulator_probe():  # pragma: no cover - compiled by CUDA-Q
        q = cudaq.qvector(1)
        x(q[0])
    # A real qpp-cpu execution, independent of the classical score optimization.
    state = cudaq.get_state(simulator_probe)
    if abs(complex(state[1])) < 0.9:
        raise RuntimeError("CUDA-Q qpp-cpu simulator probe failed")
    from v6_exact_search import exact_select
    selection, score, count = exact_select(list(pool), dict(scores))
    if count != 28:
        raise RuntimeError("exact candidate search did not enumerate 28 combinations")
    return list(selection), {**status, "enabled": True, "executed": True,
        "seed": SEED, "selection_method": "python-exact-enumeration",
        "simulator_test": "single-qubit-X-state", "combinations_checked": count,
        "real_qpu_connected": False, "remote_service_connected": False,
        "predictive_advantage_verified": False,
        "meaning": "local simulator capability plus exact combinatorial reference"}
