"""Exact, predeclared fair-lottery null statistics for OF-LGBM holdout audits.

This module only evaluates outcomes. It cannot make lottery draws predictable.
"""
from __future__ import annotations
from math import comb, sqrt


def hit_probabilities(maximum: int, count: int) -> dict[str, float]:
    if not (0 < count <= maximum):
        raise ValueError("invalid lottery configuration")
    denominator = comb(maximum, count)
    return {
        str(k): comb(count, k) * comb(maximum - count, count - k) / denominator
        if 0 <= count-k <= maximum-count else 0.0
        for k in range(count + 1)
    }


def binomial_tail(trials: int, successes: int, probability: float) -> float:
    """Exact P[Binomial(trials, probability) >= successes], inclusive."""
    if not (trials >= 0 and 0 <= successes <= trials and 0 <= probability <= 1):
        raise ValueError("invalid binomial arguments")
    return min(1.0, sum(comb(trials, k) * probability**k *
                         (1 - probability)**(trials - k)
                         for k in range(successes, trials + 1)))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if not (trials > 0 and 0 <= successes <= trials and z > 0):
        raise ValueError("invalid interval arguments")
    observed = successes / trials
    divisor = 1 + z*z/trials
    centre = (observed + z*z/(2*trials))/divisor
    radius = (z/divisor)*sqrt(observed*(1-observed)/trials + z*z/(4*trials*trials))
    return [max(0., centre-radius), min(1., centre+radius)]


def summarize_outcomes(outcomes, maximum: int, count: int, second_max=None):
    n = len(outcomes)
    if n < 1:
        raise ValueError("at least one outcome required")
    probabilities = hit_probabilities(maximum, count)
    hits = [int(r["matches"]) for r in outcomes]
    if any(k < 0 or k > count for k in hits):
        raise ValueError("invalid hit count")
    wins = sum(k >= 3 for k in hits)
    p3 = sum(probabilities[str(k)] for k in range(3, count+1))
    summary = {
        "fair_hit_distribution": probabilities,
        "fair_expected_mean_matches_per_draw": count*count/maximum,
        "observed_mean_matches_per_draw": sum(hits)/n,
        "fair_expected_3plus_count": n*p3,
        "observed_3plus_count": wins,
        "observed_3plus_rate_wilson_95pct": wilson_interval(wins, n),
        "fair_3plus_binomial_upper_tail_p": binomial_tail(n, wins, p3),
        "one_sided_p_not_search_adjusted": True,
        "statistically_verified_predictive_advantage": False,
    }
    if second_max is not None:
        if second_max < 1:
            raise ValueError("invalid second zone")
        matched = []
        for row in outcomes:
            pick, actual = row["second_pick"], row["second_actual"]
            if pick is None or actual is None or not (1 <= pick <= second_max and 1 <= actual <= second_max):
                raise ValueError("missing or invalid second zone")
            matched.append(pick == actual)
        second_hits = sum(matched)
        summary["power_second_zone"] = {
            "matched_draws": second_hits,
            "match_rate": second_hits/n,
            "fair_baseline_rate": 1/second_max,
            "fair_expected_matches": n/second_max,
            "wilson_95pct": wilson_interval(second_hits, n),
            "binomial_upper_tail_p": binomial_tail(n, second_hits, 1/second_max),
            "predictive_advantage_verified": False,
        }
    return summary
