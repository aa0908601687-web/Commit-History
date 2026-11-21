"""Isolated end-to-end adapter: model scores -> 8-number pool -> exact 6 + second area.

This adapter accepts already computed, pre-draw per-number feature rows.
It does not implement the 100 feature groups, historical CSV loader, or
walk-forward validation; those remain separate integration requirements.
"""
from __future__ import annotations

from math import isfinite

from exact_search import select_six_from_eight


def select_power_lotto_from_scores(first_area_scores, second_area_scores):
    """Choose 8 highest-ranked first-area numbers, exact 6, and second-area top 1."""
    if set(first_area_scores) != set(range(1, 39)):
        raise ValueError("First-area scores must cover exactly numbers 1..38")
    if set(second_area_scores) != set(range(1, 9)):
        raise ValueError("Second-area scores must cover exactly numbers 1..8")
    for score in list(first_area_scores.values()) + list(second_area_scores.values()):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
            raise ValueError("Scores must be finite numeric values")
    candidates = sorted(first_area_scores, key=lambda n: (-first_area_scores[n], n))[:8]
    selected = select_six_from_eight({n: first_area_scores[n] for n in candidates})
    second = min(second_area_scores, key=lambda n: (-second_area_scores[n], n))
    return {
        "first_area": selected["numbers"],
        "second_area": second,
        "candidate_pool": tuple(sorted(candidates)),
        "objective_score": selected["score"],
    }


def select_from_model_probabilities(first_numbers, first_probabilities, second_numbers, second_probabilities):
    """Bridge a model's positive-class probabilities into deterministic selection.

    All model outputs must be generated from pre-draw information only.
    """
    if len(first_numbers) != len(first_probabilities) or len(second_numbers) != len(second_probabilities):
        raise ValueError("Numbers and probabilities must have matching lengths")
    if len(set(first_numbers)) != len(first_numbers) or len(set(second_numbers)) != len(second_numbers):
        raise ValueError("Duplicate numbers are not allowed")
    return select_power_lotto_from_scores(
        dict(zip(first_numbers, first_probabilities)),
        dict(zip(second_numbers, second_probabilities)),
    )
