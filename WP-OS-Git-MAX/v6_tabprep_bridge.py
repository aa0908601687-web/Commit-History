#!/usr/bin/env python3
"""Optional V6 TabPrep + LightGBM feature scorer.

Upstream: https://github.com/atschalz/tabprep (Apache-2.0).
This module does NOT vendor upstream code or silently fall back to OpenFE.
Training uses completed past draws only; each prediction feature uses <=10 prior
draws even when older draws supply training anchors. No lottery advantage claim.
"""
from __future__ import annotations

import math

from v6_openfe_bridge import SEED, VIEW_LIMITS, _base_row, _training_frame


def score_three_history_views_tabprep(rows):
    if len(rows) < 12:
        raise ValueError("TabPrep needs at least 12 completed draws")
    try:
        import pandas as pd
        from lightgbm import LGBMClassifier
        from tabprep import TabPrepFeatureGenerator
    except ImportError as exc:
        raise RuntimeError(
            "TabPrep is optional and not installed. Follow docs/V6_TABPREP_SETUP.md "
            "to install upstream TabPrep, AutoGluon, pandas, and LightGBM."
        ) from exc

    train_x, train_y = _training_frame(rows, three_views=True)
    current_x = pd.DataFrame(
        [_base_row(rows, number, limit) for limit in VIEW_LIMITS.values()
         for number in range(1, 39)]
    )

    # Disable out-of-fold target encoding for an initial conservative baseline.
    # Do not use future outcomes or fit the transformer on current rows.
    generator = TabPrepFeatureGenerator(
        target_type="binary", random_state=SEED, use_oofte=False
    )
    train_features = generator.fit_transform(train_x, train_y)
    current_features = generator.transform(current_x)
    if list(train_features.columns) != list(current_features.columns):
        raise ValueError("TabPrep train/inference schema mismatch")

    model = LGBMClassifier(
        n_estimators=80, learning_rate=0.05, max_depth=4, num_leaves=15,
        random_state=SEED, n_jobs=1, verbosity=-1,
    )
    model.fit(train_features, train_y)
    predictions = model.predict_proba(current_features)[:, 1]
    if len(predictions) != 38 * len(VIEW_LIMITS):
        raise ValueError("TabPrep returned wrong number of predictions")
    if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in predictions):
        raise ValueError("TabPrep emitted invalid probability")

    by_view = {}
    for view_idx, name in enumerate(VIEW_LIMITS):
        by_view[name] = {
            number: float(predictions[view_idx * 38 + number - 1])
            for number in range(1, 39)
        }
    fused = {
        number: sum(by_view[name][number] for name in VIEW_LIMITS) / len(VIEW_LIMITS)
        for number in range(1, 39)
    }
    return fused, {
        "engine": "TabPrep+LightGBM",
        "experimental": True,
        "tabprep_executed": True,
        "feature_window": [5, 10],
        "full_archive_role": "training_anchors_only",
        "history_views": list(VIEW_LIMITS),
        "per_view_scores": by_view,
        "generated_feature_columns": int(train_features.shape[1]),
        "seed": SEED,
        "leakage_safe_design": True,
        "predictive_advantage_verified": False,
        "upstream": "https://github.com/atschalz/tabprep",
    }
