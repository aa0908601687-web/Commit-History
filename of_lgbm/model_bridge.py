"""Explicit research bridge from trained OpenFE/LightGBM models to lottery selection.

Callers must supply chronologically separated training/validation features.
Each validation frame must contain one row for every legal number of its area,
ordered by the accompanying number list. This is not a CSV ingestion pipeline.
"""
from __future__ import annotations

from integration import select_from_model_probabilities
from pipeline import train_openfe_lightgbm


def _positive_class_probabilities(output):
    """Extract positive class from sklearn-compatible (N, 2) probabilities."""
    if len(output) == 0:
        raise ValueError("No model probabilities returned")
    values = []
    for row in output:
        if len(row) != 2:
            raise ValueError("Expected binary class probabilities")
        value = float(row[1])
        if not 0.0 <= value <= 1.0:
            raise ValueError("Probabilities must be in [0, 1]")
        values.append(value)
    return values


def train_and_select(first_train_x, first_train_y, first_valid_x, first_valid_y,
                     second_train_x, second_train_y, second_valid_x, second_valid_y,
                     *, first_numbers, second_numbers, seed=42, n_jobs=1,
                     trainer=train_openfe_lightgbm):
    """Train two isolated area models and feed validation scores to exact search.

    Validation labels are used only for the existing trainer's input contract;
    they must never be used to create features or to choose numbers.
    No claim of out-of-sample evaluation is made by this adapter.
    """
    if tuple(sorted(first_numbers)) != tuple(range(1, 39)):
        raise ValueError("First-area validation numbers must be a permutation of 1..38")
    if tuple(sorted(second_numbers)) != tuple(range(1, 9)):
        raise ValueError("Second-area validation numbers must be a permutation of 1..8")
    if len(first_valid_x) != 38 or len(second_valid_x) != 8:
        raise ValueError("Validation features must contain one row per legal number")
    first = trainer(first_train_x, first_train_y, first_valid_x, first_valid_y,
                    seed=seed, n_jobs=n_jobs)
    second = trainer(second_train_x, second_train_y, second_valid_x, second_valid_y,
                     seed=seed, n_jobs=n_jobs)
    first_probs = _positive_class_probabilities(first["probabilities"])
    second_probs = _positive_class_probabilities(second["probabilities"])
    return select_from_model_probabilities(
        first_numbers, first_probs, second_numbers, second_probs
    )
