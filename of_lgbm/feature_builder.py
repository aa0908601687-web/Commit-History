"""Leakage-safe per-number feature construction for three historical views.

This baseline builder does not replace the 100 independent feature sets or
claim official-data provenance. Features use only draws before each target.
"""
from __future__ import annotations

from history_adapter import historical_windows


def build_number_features(rows, target_index, area):
    """Return number-indexed rows for a target draw, excluding that draw.

    target_index may equal len(rows) for a future draw. All views are
    counts/rates, not arbitrary weights. No target outcomes are inspected.
    """
    if area not in ("first", "second"):
        raise ValueError("area must be first or second")
    windows = historical_windows(rows, target_index)
    maximum = 38 if area == "first" else 8
    features = []
    for number in range(1, maximum + 1):
        record = {"number": number}
        for key in ("last_10", "last_20", "all_history"):
            draws = windows[key]
            hits = sum(
                number in draw["first"] if area == "first"
                else number == draw["second"]
                for draw in draws
            )
            record[f"{key}_hits"] = hits
            record[f"{key}_rate"] = hits / len(draws) if draws else 0.0
        features.append(record)
    return features


def build_supervised_history(rows, target_indices, area):
    """Build historical training rows with labels from past, known draws.

    A future target (index == len(rows)) is never allowed in training.
    """
    if area not in ("first", "second"):
        raise ValueError("area must be first or second")
    result_x, result_y = [], []
    for index in target_indices:
        if not 0 < index < len(rows):
            raise ValueError("Training targets must be historical and have prior draws")
        features = build_number_features(rows, index, area)
        actual = rows[index]
        for record in features:
            number = record["number"]
            result_x.append(record)
            result_y.append(int(
                number in actual["first"] if area == "first"
                else number == actual["second"]
            ))
    return result_x, result_y
