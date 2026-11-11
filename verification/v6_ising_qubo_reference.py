"""CPU reference for V6 OF3Q 8-to-6 Ising/QUBO objective.

No NVIDIA solver, exact, or predictive model is invoked here.
All scores are supplied by the upstream caller.
"""
from itertools import combinations
from math import isfinite

def build_qubo(scores, pair_scores=None, penalty=None):
    """Minimize E=-sum(s_i*x_i)-sum(p_ij*x_i*x_j)+A*(sum(x)-6)^2.

    Return upper-triangular Q with E=sum(Q[i,j]*x_i*x_j)+offset.
    """
    if len(scores) != 8 or any(not isfinite(float(s)) for s in scores):
        raise ValueError("exactly eight finite scores required")
    pairs = pair_scores or {}
    for (i, j), value in pairs.items():
        if not (0 <= i < j < 8) or not isfinite(float(value)):
            raise ValueError("invalid pair index or score")
    magnitude = sum(abs(float(s)) for s in scores) + sum(abs(float(v)) for v in pairs.values())
    A = float(penalty) if penalty is not None else magnitude + 1.0
    if not isfinite(A) or A <= magnitude:
        raise ValueError("penalty must exceed sum of absolute rewards")
    Q = {(i, i): -float(scores[i]) - 11*A for i in range(8)}
    for i, j in combinations(range(8), 2):
        Q[i, j] = 2*A - float(pairs.get((i, j), 0.0))
    return Q, 36*A

def energy(bits, Q, offset):
    if len(bits) != 8 or any(b not in (0, 1) for b in bits):
        raise ValueError("eight binary bits required")
    return offset + sum(value*bits[i]*bits[j] for (i, j), value in Q.items())

def exact_six(scores, pair_scores=None):
    Q, offset = build_qubo(scores, pair_scores)
    ranked = []
    for selected in combinations(range(8), 6):
        bits = tuple(int(i in selected) for i in range(8))
        ranked.append((energy(bits, Q, offset), selected))
    best = min(ranked, key=lambda item: (round(item[0], 10), item[1]))
    return {"indices": best[1], "energy": best[0], "combinations_checked": len(ranked)}

def check_solver_output(bits, scores, pair_scores=None, atol=1e-8):
    Q, offset = build_qubo(scores, pair_scores)
    if len(bits) != 8 or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        return {"valid": False, "optimal": False}
    if sum(bits) != 6:
        return {"valid": False, "optimal": False}
    exact = exact_six(scores, pair_scores)
    value = energy(bits, Q, offset)
    return {"valid": True, "optimal": abs(value-exact["energy"]) <= atol,
            "energy": value, "reference": exact}
