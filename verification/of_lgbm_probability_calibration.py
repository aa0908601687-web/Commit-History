"""Causal, research-only calibration audit of per-number OF-LGBM probabilities.

Each historical draw is assessed using a coefficient chosen ONLY from previously
assessed draws. A short five-fold holdout cannot justify fitted calibration.
This measures probability error, NOT lottery winning probability improvement.
"""
from __future__ import annotations
import math

GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_PRIOR_DRAWS = 30


def validate(scores, actual, maximum, count):
    if len(scores) != maximum or len(actual) != count or len(set(actual)) != count:
        raise ValueError("invalid score vector or draw")
    if any(not isinstance(n, int) or not 1 <= n <= maximum for n in actual):
        raise ValueError("invalid draw number")
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
           for p in scores):
        raise ValueError("nonfinite or out-of-range prediction probability")


def brier(scores, actual):
    winners = set(actual)
    return sum((float(p) - int(i in winners)) ** 2
               for i, p in enumerate(scores, 1)) / len(scores)


def shrink(scores, fair, alpha):
    return [fair + alpha * (float(p) - fair) for p in scores]


def choose_alpha(prior, maximum, count):
    """Tune solely from prior completed predictions; never from the target draw."""
    if len(prior) < MIN_PRIOR_DRAWS:
        return 0.0
    fair = count / maximum
    losses = {}
    for a in GRID:
        losses[a] = sum(brier(shrink(r["scores"], fair, a), r["actual"])
                        for r in prior) / len(prior)
    # Must outperform fairness on past predictions before enabling a model signal.
    winner = min(GRID, key=lambda a: (losses[a], a))
    if losses[winner] >= losses[0.0] - 0.0001:
        return 0.0
    return winner


def audit_predictions(records, maximum, count):
    """Prequential audit: each chosen alpha sees only earlier outcomes."""
    if not records:
        raise ValueError("no predictions to audit")
    if not 0 < count <= maximum:
        raise ValueError("invalid lottery configuration")
    prior, rows = [], []
    fair = count / maximum
    for r in records:
        scores = list(r["scores"])
        actual = list(r["actual"])
        validate(scores, actual, maximum, count)
        alpha = choose_alpha(prior, maximum, count)
        calibrated = shrink(scores, fair, alpha)
        rows.append({
            "prior_predictions": len(prior),
            "alpha_from_prior_only": alpha,
            "raw_brier": brier(scores, actual),
            "calibrated_brier": brier(calibrated, actual),
            "fair_brier": brier([fair] * maximum, actual),
        })
        # Current draw is eligible only for FUTURE predictions.
        prior.append({"scores": scores, "actual": actual})
    def average(key):
        return sum(r[key] for r in rows) / len(rows)
    return {
        "method": "causal_prequential_shrinkage_to_fair_per_number_probability",
        "number_of_draws": len(rows),
        "minimum_prior_predictions_before_fitting": MIN_PRIOR_DRAWS,
        "calibration_fitted_draws": sum(r["prior_predictions"] >= MIN_PRIOR_DRAWS for r in rows),
        "raw_brier": average("raw_brier"),
        "calibrated_brier": average("calibrated_brier"),
        "fair_brier": average("fair_brier"),
        "calibration_verified_improvement": False,
        "predictive_advantage_verified": False,
        "not_used_to_reselect_test_tickets": True,
        "status": "insufficient_prior_predictions" if len(rows) <= MIN_PRIOR_DRAWS
                  else "exploratory_prequential_audit_not_independent_validation",
        "per_draw": rows,
    }
