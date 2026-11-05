#!/usr/bin/env python3
"""Leakage-safe OpenFE bridges for V6 first-zone ranking.

OpenFE generates tabular features; LightGBM converts them to per-number scores.
All training rows use only draws completed before their target draw.
"""
from __future__ import annotations

import random

SEED = 20261103
VIEW_LIMITS = {"latest_5": 5, "latest_10": 10, "full_archive": None}


def _base_row(history, number, view_limit=None):
    view = history[-10:] if view_limit is None else history[-view_limit:]
    # Full archive supplies historical training anchors, never >10-draw features.
    five = history[-5:]
    ten = history[-10:]
    last_seen = next(
        (age for age, draw in enumerate(reversed(view), 1) if number in draw[1]),
        len(view) + 1,
    )
    # Full-archive is a training-anchor view; its current statistics use a
    # distinct time-decayed 10-draw signal rather than duplicating latest_10.
    weighted = sum((i + 1) * (number in draw[1]) for i, draw in enumerate(view))
    weighted_denominator = sum(range(1, len(view) + 1))
    view_frequency = (weighted / weighted_denominator) if view_limit is None else (sum(number in draw[1] for draw in view) / len(view))
    return {
        "number": float(number),
        "number_scaled": number / 38.0,
        "freq_5": float(sum(number in draw[1] for draw in five)),
        "freq_10": float(sum(number in draw[1] for draw in ten)),
        "freq_view": float(view_frequency),
        "gap_view": float(min(last_seen, 10)),
        "view_rows": float(len(view)),
        "view_is_10": float(view_limit == 10),
        "view_is_5": float(view_limit == 5),
        "view_is_full": float(view_limit is None),
        "odd": float(number % 2),
        "low_zone": float(number <= 19),
    }


def _training_frame(rows, *, three_views=False, max_anchors=80):
    import pandas as pd

    samples, labels = [], []
    limits = VIEW_LIMITS.values() if three_views else (None,)
    ends = list(range(10, len(rows)))
    if three_views and len(ends) > max_anchors:
        positions = {
            round(index * (len(ends) - 1) / (max_anchors - 1))
            for index in range(max_anchors)
        }
        ends = [ends[index] for index in sorted(positions)]
    for end in ends:
        history = rows[:end]
        target = set(rows[end][1])
        for view_limit in limits:
            if view_limit is not None and len(history) < view_limit:
                continue
            for number in range(1, 39):
                samples.append(_base_row(history, number, view_limit))
                labels.append(int(number in target))
    if len(set(labels)) != 2:
        raise ValueError("OpenFE bridge requires both positive and negative labels")
    return pd.DataFrame(samples), pd.Series(labels, name="selected")


def _fit_and_score(train_x, train_y, current_x, *, n_jobs, max_features):
    import numpy as np
    from lightgbm import LGBMClassifier
    from openfe import OpenFE, transform

    random.seed(SEED)
    np.random.seed(SEED)
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
    return probabilities, len(generated), len(selected)


def score_first_zone(rows, *, n_jobs=1, max_features=12):
    """Backward-compatible full-archive OpenFE score."""
    if len(rows) < 12:
        raise ValueError("at least 12 completed draws required")
    import pandas as pd

    train_x, train_y = _training_frame(rows)
    current_x = pd.DataFrame(
        [_base_row(rows, number, None) for number in range(1, 39)]
    )
    probabilities, generated, used = _fit_and_score(
        train_x, train_y, current_x, n_jobs=n_jobs, max_features=max_features
    )
    return {
        number: float(probabilities[number - 1]) for number in range(1, 39)
    }, {
        "engine": "OpenFE+LightGBM",
        "feature_window": [5, 10],
        "generated_features": generated,
        "used_features": used,
        "seed": SEED,
        "leakage_safe": True,
        "history_views": ["full_archive"],
    }


def score_three_history_views(rows, *, n_jobs=1, max_features=12):
    """Score 5/10/full-archive-training views. All prediction features use <=10 prior draws."""
    if len(rows) < 12:
        raise ValueError("three-view OpenFE requires at least 12 completed draws")
    import pandas as pd

    train_x, train_y = _training_frame(rows, three_views=True)
    current_records = []
    for _, view_limit in VIEW_LIMITS.items():
        current_records.extend(
            _base_row(rows, number, view_limit) for number in range(1, 39)
        )
    current_x = pd.DataFrame(current_records)
    probabilities, generated, used = _fit_and_score(
        train_x, train_y, current_x, n_jobs=n_jobs, max_features=max_features
    )

    by_view = {}
    offset = 0
    for view_name in VIEW_LIMITS:
        by_view[view_name] = {
            number: float(probabilities[offset + number - 1])
            for number in range(1, 39)
        }
        offset += 38
    assert len(probabilities) == 38 * len(VIEW_LIMITS), "view-score alignment mismatch"
    fused = {
        number: sum(by_view[name][number] for name in VIEW_LIMITS) / len(VIEW_LIMITS)
        for number in range(1, 39)
    }
    return fused, {
        "feature_window": [5, 10],
        "v3_cache_features_used": False,
        "full_archive_role": "training_anchors_only",
        "engine": "OpenFE+LightGBM",
        "mode": "three_history_views_100_percent",
        "generated_features": generated,
        "used_features": used,
        "seed": SEED,
        "leakage_safe": True,
        "history_views": list(VIEW_LIMITS),
        "view_weights": {name: 1 / 3 for name in VIEW_LIMITS},
        "per_view_scores": by_view,
        "training_anchor_count": len(train_x) // (38 * len(VIEW_LIMITS)),
        "training_anchor_strategy": "uniform_across_full_archive",
    }
