#!/usr/bin/env python3
"""Leakage-safe OpenFE feature bridge for V6 first-zone ranking.

OpenFE generates tabular features; LightGBM converts them to per-number scores.
Every training row only uses draws completed before its target draw.
"""
from __future__ import annotations

from collections import Counter
import random

SEED = 20261103


def _base_row(history, number):
    five = history[-5:]
    ten = history[-10:]
    last_seen = next(
        (age for age, draw in enumerate(reversed(history), 1) if number in draw[1]),
        len(history) + 1,
    )
    return {
        "number": float(number),
        "number_scaled": number / 38.0,
        "freq_5": float(sum(number in draw[1] for draw in five)),
        "freq_10": float(sum(number in draw[1] for draw in ten)),
        "gap": float(min(last_seen, 20)),
        "odd": float(number % 2),
        "low_zone": float(number <= 19),
    }


def _training_frame(rows):
    import pandas as pd

    samples, labels = [], []
    for end in range(10, len(rows)):
        history = rows[:end]
        target = set(rows[end][1])
        for number in range(1, 39):
            samples.append(_base_row(history, number))
            labels.append(int(number in target))
    if len(set(labels)) != 2:
        raise ValueError("OpenFE bridge requires both positive and negative labels")
    return pd.DataFrame(samples), pd.Series(labels, name="selected")


def score_first_zone(rows, *, n_jobs=1, max_features=12):
    """Return deterministic OpenFE+LightGBM probabilities for numbers 1..38."""
    if len(rows) < 12:
        raise ValueError("at least 12 completed draws required")

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from openfe import OpenFE, transform

    random.seed(SEED)
    np.random.seed(SEED)
    train_x, train_y = _training_frame(rows)
    current_x = pd.DataFrame([_base_row(rows, number) for number in range(1, 39)])

    generator = OpenFE()
    generated = generator.fit(
        data=train_x,
        label=train_y,
        task="classification",
        n_jobs=n_jobs,
        verbose=False,
    )
    selected = generated[:max_features]
    if selected:
        train_augmented, current_augmented = transform(
            train_x, current_x, selected, n_jobs=n_jobs
        )
    else:
        train_augmented, current_augmented = train_x, current_x

    model = LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=SEED,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(train_augmented, train_y)
    probabilities = model.predict_proba(current_augmented)[:, 1]
    return {
        number: float(probabilities[number - 1]) for number in range(1, 39)
    }, {
        "engine": "OpenFE+LightGBM",
        "generated_features": len(generated),
        "used_features": len(selected),
        "seed": SEED,
        "leakage_safe": True,
    }
