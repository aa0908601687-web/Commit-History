"""Exact first-zone optimization over a supplied candidate pool; no predictive claim."""
from itertools import combinations
from math import comb

def exact_select(pool, scores, size=6):
    pool = sorted(set(pool))
    if len(pool) < size:
        raise ValueError("candidate pool too small")
    if any(n not in scores for n in pool):
        raise ValueError("missing candidate score")
    best = max(combinations(pool, size), key=lambda c: (sum(scores[n] for n in c), tuple(-n for n in c)))
    return list(best), sum(scores[n] for n in best), comb(len(pool), size)

def compare_selection(pool, scores, approximate, size=6):
    exact, optimum, count = exact_select(pool, scores, size)
    approx = sorted(approximate)
    if len(approx) != size or len(set(approx)) != size or not set(approx) <= set(pool):
        raise ValueError("invalid approximate selection")
    approx_score = sum(scores[n] for n in approx)
    return {"exact_selection": exact, "exact_score": optimum,
            "approximate_selection": approx, "approximate_score": approx_score,
            "score_gap": max(0.0, optimum - approx_score),
            "searched_combinations": count, "globally_optimal_within_pool": True,
            "predictive_advantage_verified": False}
