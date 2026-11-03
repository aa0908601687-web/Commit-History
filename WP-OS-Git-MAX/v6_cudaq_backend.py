"""Optional NVIDIA CUDA-Q local simulator backend for V6.

This module never selects a remote target or a real QPU.  It uses CUDA-Q's
local qpp-cpu statevector target and a deterministic score-phase ansatz.  The
six-of-eight cardinality rule is enforced by classical post-selection.
"""
from __future__ import annotations

import importlib
from typing import Mapping, Sequence


TARGET = "qpp-cpu"
SEED = 20261103


def availability() -> dict:
    """Return dependency status without importing CUDA-Q at module import time."""
    try:
        cudaq = importlib.import_module("cudaq")
    except (ImportError, OSError) as exc:
        return {
            "installed": False,
            "backend": TARGET,
            "qpu": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "installed": True,
        "backend": TARGET,
        "qpu": False,
        "version": getattr(cudaq, "__version__", "unknown"),
    }


def _normalise(pool: Sequence[int], scores: Mapping[int, float]) -> list[float]:
    values = [float(scores[number]) for number in pool]
    low, high = min(values), max(values)
    span = high - low
    if not span:
        return [0.5] * len(values)
    return [(value - low) / span for value in values]


def select_six(
    pool: Sequence[int],
    scores: Mapping[int, float],
    *,
    required: bool = False,
) -> tuple[list[int] | None, dict]:
    """Run the local CUDA-Q ansatz and return six numbers from an eight-number pool."""
    if len(pool) != 8 or len(set(pool)) != 8:
        raise ValueError("CUDA-Q backend requires eight distinct candidates")
    if any(number not in scores for number in pool):
        raise ValueError("missing CUDA-Q candidate score")

    status = availability()
    if not status["installed"]:
        if required:
            raise RuntimeError("CUDA-Q is required but unavailable: " + status["reason"])
        return None, {
            **status,
            "enabled": True,
            "executed": False,
            "fallback": "existing local Python statevector",
        }

    cudaq = importlib.import_module("cudaq")
    cudaq.set_target(TARGET)
    if hasattr(cudaq, "set_random_seed"):
        cudaq.set_random_seed(SEED)

    @cudaq.kernel
    def score_phase(weights: list[float], gamma: float, beta: float):  # pragma: no cover - compiled by CUDA-Q
        qubits = cudaq.qvector(8)
        h(qubits)
        for index in range(8):
            rz(2.0 * gamma * weights[index], qubits[index])
            rx(2.0 * beta, qubits[index])

    weights = _normalise(pool, scores)
    best = None
    for step in range(1, 17):
        gamma = step * 3.141592653589793 / 32.0
        beta = step * 3.141592653589793 / 64.0
        state = cudaq.get_state(score_phase, weights, gamma, beta)
        for mask in range(256):
            if mask.bit_count() != 6:
                continue
            probability = abs(complex(state[mask])) ** 2
            selected = tuple(sorted(pool[index] for index in range(8) if mask >> index & 1))
            candidate = (probability, sum(scores[number] for number in selected), selected)
            if best is None or candidate > best:
                best = candidate

    selected = list(best[2])
    return selected, {
        **status,
        "enabled": True,
        "executed": True,
        "seed": SEED,
        "ansatz": "8-qubit score-phase p=1 with six-of-eight classical post-selection",
        "real_qpu_connected": False,
        "remote_service_connected": False,
        "predictive_advantage_verified": False,
        "meaning": "local numerical research backend; not lottery hit-rate improvement",
    }
