"""Causal, bounded 8-number-pool combination scoring for standalone OF-LGBM.

Pair signals are exploratory, strongly shrunk, and do NOT imply predictive advantage.
Only draws completed before the target may be passed as history.
"""
from itertools import combinations
import math

PAIR_WEIGHT = 0.05

def select_ticket(scores, history, count, maximum, pool_size=8):
    if len(scores) != maximum or len(history) < 20:
        raise ValueError("invalid scores or insufficient prior history")
    if not 1 <= count <= pool_size <= maximum:
        raise ValueError("invalid combination configuration")
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("nonfinite number scores")
    ordered = sorted(range(1, maximum + 1), key=lambda n: (-float(scores[n - 1]), n))
    pool = ordered[:pool_size]
    recent10, recent20 = history[-10:], history[-20:]
    prior = count * (count - 1) / (maximum * (maximum - 1))
    pair_values = {}
    for a, b in combinations(pool, 2):
        h10 = sum(a in row and b in row for row in recent10)
        h20 = sum(a in row and b in row for row in recent20)
        # Ten pseudo-observations suppress one-off coincidental pair spikes.
        p10 = (h10 + 10 * prior) / 20
        p20 = (h20 + 10 * prior) / 30
        pair_values[(min(a, b), max(a, b))] = 0.5 * (p10 + p20)
    options = list(combinations(pool, count))
    def value(ticket):
        base = sum(float(scores[n - 1]) for n in ticket) / count
        pairs = list(combinations(sorted(ticket), 2))
        pair = sum(pair_values[p] for p in pairs) / len(pairs) if pairs else 0.0
        return (1 - PAIR_WEIGHT) * base + PAIR_WEIGHT * pair
    chosen = max(options, key=lambda ticket: (value(ticket), tuple(-n for n in sorted(ticket))))
    return tuple(sorted(chosen)), {
        "pool_size": pool_size, "candidate_combinations": len(options),
        "selection_objective": "95pct_mean_lgbm_score_plus_5pct_shrunk_recent_pair_rate",
        "pair_weight": PAIR_WEIGHT,
        "pair_windows": [10, 20],
        "pair_signal_verified_predictive": False,
        "search_scope": "exact within top-eight pool only",
    }
