#!/usr/bin/env python3
"""Local numerical-fidelity verification for the V6 QAOA-inspired solver.

This is a deterministic statevector cross-check, not NVIDIA/JUPITER access,
not a QPU, and not evidence of lottery prediction accuracy.
"""
from __future__ import annotations

import math


def _statevector(costs, gamma, beta, *, quantize_digits=None):
    size = len(costs)
    qubits = int(math.log2(size))
    if 2**qubits != size:
        raise ValueError("cost vector length must be a power of two")
    state = [complex(math.cos(-gamma*c), math.sin(-gamma*c))/math.sqrt(size) for c in costs]
    for bit in range(qubits):
        cosine, sine = math.cos(beta), -1j*math.sin(beta)
        for index in range(size):
            if not index >> bit & 1:
                other = index | (1 << bit)
                left, right = state[index], state[other]
                state[index] = cosine*left + sine*right
                state[other] = sine*left + cosine*right
        if quantize_digits is not None:
            state = [complex(round(x.real, quantize_digits), round(x.imag, quantize_digits)) for x in state]
    return state


def _fidelity(reference, candidate):
    ref_norm = sum(abs(x)**2 for x in reference)
    cand_norm = sum(abs(x)**2 for x in candidate)
    if ref_norm <= 0 or cand_norm <= 0:
        raise ValueError("statevector norm must be positive")
    overlap = sum(a.conjugate()*b for a, b in zip(reference, candidate))
    return abs(overlap)**2/(ref_norm*cand_norm), cand_norm


def verified_statevector(costs, gamma, beta, *, minimum_fidelity=0.99999):
    """Cross-check double precision against an independently rounded path."""
    if not 0 < minimum_fidelity <= 1:
        raise ValueError("minimum_fidelity must be in (0, 1]")
    reference = _statevector(costs, gamma, beta)
    candidate = _statevector(costs, gamma, beta, quantize_digits=12)
    fidelity, norm = _fidelity(reference, candidate)
    if abs(sum(abs(x)**2 for x in reference)-1.0) > 1e-10:
        raise RuntimeError("reference statevector normalization failed")
    if fidelity < minimum_fidelity:
        raise RuntimeError(f"local numerical fidelity {fidelity:.12f} below threshold")
    return candidate, {"fidelity": fidelity, "state_norm": norm}

