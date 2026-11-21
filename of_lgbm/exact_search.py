"""Deterministic 8-candidate / 6-selection adapter for isolated research.

This module does not assert that lottery outcomes are predictable. Scores must
be produced using only information available before the target drawing.
"""
from __future__ import annotations

from itertools import combinations
from math import isfinite


def select_six_from_eight(candidate_scores):
    """Select the highest-scoring six distinct numbers from exactly eight.

    Exhaustively evaluates all C(8, 6) = 28 combinations. The objective is
    the sum of per-number scores, with lexicographic tie breaking. This is
    an exact solver for this *additive* objective only; it is not an
    implementation of the full V6 scoring model.
    """
    if len(candidate_scores) != 8:
        raise ValueError("Exactly eight candidate scores are required")
    if any(
        not isinstance(number, int) or isinstance(number, bool)
        or not 1 <= number <= 38
        for number in candidate_scores
    ):
        raise ValueError("Candidate numbers must be integers in 1..38")
    if any(
        isinstance(score, bool) or not isinstance(score, (int, float))
        or not isfinite(score)
        for score in candidate_scores.values()
    ):
        raise ValueError("Scores must be finite numeric values")
    best = min(
        combinations(sorted(candidate_scores), 6),
        key=lambda group: (-sum(candidate_scores[n] for n in group), group),
    )
    return {"numbers": best, "score": sum(candidate_scores[n] for n in best)}
