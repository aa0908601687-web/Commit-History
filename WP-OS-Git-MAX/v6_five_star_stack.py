"""Optional V6 OF3Q high-performance stack probes; no replacement of production E2E.

Run: python WP-OS-Git-MAX/v6_five_star_stack.py
Optional libraries are probed only if installed. Results are capabilities, not speed rankings.
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import time

LIBRARIES = {
    "featuretools": "featuretools",
    "autogluon": "autogluon",
    "tsfresh": "tsfresh",
    "polars": "polars",
    "jax": "jax",
    "numpy": "numpy",
    "numba": "numba",
    "ortools": "ortools",
    "gurobipy": "gurobipy",
    "pyscipopt": "pyscipopt",
    "lightgbm": "lightgbm",
}

def availability():
    return {name: importlib.util.find_spec(module) is not None
            for name, module in LIBRARIES.items()}

def exact_six(pool, scores):
    """Independent stdlib reference for exactly 8 choose 6 candidates."""
    if len(pool) != 8 or len(set(pool)) != 8:
        raise ValueError("expected eight distinct first-zone candidates")
    if any(not 1 <= n <= 38 for n in pool):
        raise ValueError("first-zone candidate out of range")
    if set(scores) != set(pool):
        raise ValueError("scores must cover precisely the candidate pool")
    choices = list(itertools.combinations(sorted(pool), 6))
    best = max(choices, key=lambda c: (sum(scores[n] for n in c), tuple(-n for n in c)))
    return list(best), sum(scores[n] for n in best), len(choices)

def ortools_six(pool, scores):
    """Optional CP-SAT path, separate from stdlib enumeration.

    Integer score scaling makes this a separate approximate-score objective;
    do not assert exact equality when input scores are near ties.
    """
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    x = {n: model.new_bool_var(f"pick_{n}") for n in pool}
    model.add(sum(x.values()) == 6)
    scale = 1_000_000
    model.maximize(sum(round(scores[n] * scale) * x[n] for n in pool))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("OR-Tools did not certify optimality")
    selection = sorted(n for n in pool if solver.value(x[n]))
    return selection, sum(scores[n] for n in selection)

def run():
    pool = [9, 15, 7, 27, 18, 19, 32, 24]
    scores = {n: float(39-n) / 38 for n in pool}  # synthetic, not lottery prediction
    start = time.perf_counter()
    selection, score, checked = exact_six(pool, scores)
    elapsed_ms = (time.perf_counter() - start) * 1000
    result = {"data": "synthetic", "production_model_executed": False,
              "available": availability(), "exact_reference": {
                  "selection": selection, "score": score,
                  "combinations": checked, "elapsed_ms": elapsed_ms}}
    if result["available"]["ortools"]:
        t = time.perf_counter()
        selected, value = ortools_six(pool, scores)
        result["ortools"] = {"selection": selected, "score": value,
                             "elapsed_ms": (time.perf_counter()-t)*1000,
                             "matches_reference_score": abs(value-score)<1e-9}
    return result

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
