"""V6 local, dependency-free QAOA state-vector research prototype.

Optimizes a *supplied* score vector over a small pool; no predictive claims.
Not a real QPU. Does not connect to ABE or production WP-OS.
"""
import argparse
import cmath
import itertools
import json
import math

def solve(scores, choose=6, gamma=0.65, beta=0.35):
    n=len(scores)
    if not 1 <= choose <= n <= 10:
        raise ValueError("Require 1 <= choose <= number of scores <= 10")
    if any(not math.isfinite(float(v)) for v in scores):
        raise ValueError("All scores must be finite")
    scores=list(map(float,scores))
    size=1 << n
    # Cardinality-preserving initial superposition over feasible subsets.
    valid=[sum((mask >> i)&1 for i in range(n))==choose for mask in range(size)]
    count=sum(valid)
    state=[(1/math.sqrt(count) if valid[mask] else 0j) for mask in range(size)]
    def objective(mask):
        return sum(scores[i] for i in range(n) if (mask>>i)&1)
    # Phase separator: diagonal score Hamiltonian. Remains in feasible subspace.
    state=[a*cmath.exp(1j*gamma*objective(mask)) for mask,a in enumerate(state)]
    # XY mixer: for each pair, rotate |01>, |10>; preserves Hamming weight.
    for i in range(n):
        for j in range(i+1,n):
            m1,m2=1<<i,1<<j
            for mask in range(size):
                if mask&m1 or not mask&m2:
                    continue
                other=mask^m1^m2
                a,b=state[mask],state[other]
                state[mask]=math.cos(beta)*a-1j*math.sin(beta)*b
                state[other]=math.cos(beta)*b-1j*math.sin(beta)*a
    probabilities=[abs(a)**2 for a in state]
    top=max((i for i in range(size) if valid[i]),
            key=lambda i:(probabilities[i],objective(i),-i))
    return {"selected_indices":[i for i in range(n) if top>>i&1],
            "probability":probabilities[top], "total_probability":sum(probabilities),
            "n_candidates":n,"choose":choose,"method":"one_layer_qaoa_statevector_xy",
            "real_qpu":False,"historical_backtest":False,
            "predictive_advantage_verified":False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scores",required=True,help="comma-separated supplied scores")
    parser.add_argument("--choose",type=int,default=6)
    args=parser.parse_args()
    print(json.dumps(solve([float(x) for x in args.scores.split(",")],args.choose),indent=2))

if __name__=="__main__":
    main()
