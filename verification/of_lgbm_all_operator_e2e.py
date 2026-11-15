#!/usr/bin/env python3
"""Opt-in independent OF-LGBM full-23-operator research E2E.

Runs actual OpenFE candidate screening, executes a witness for every available
operator family, feeds one witness per operator into LightGBM, and performs exact
8-choose-k selection. Never claims predictive advantage or official provenance.
The existing three-lottery E2E is NOT modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from openfe import OpenFE, get_candidate_features, transform, tree_to_formula
from openfe.FeatureGenerator import (
    all_operators, num_operators, num_num_operators,
    cat_num_operators, cat_cat_operators,
)
from lightgbm import LGBMClassifier

from openfe_three_lottery_integration import CONFIGS, read_real_csv

VIEW_ANCHORS = {"latest_5": 5, "latest_10": 10, "full_archive": None}
VIEW_WEIGHTS = {"latest_5": 0.35, "latest_10": 0.35, "full_archive": 0.30}
CATEGORY_COLUMNS = ("parity_group", "number_band", "recency_group")
EXPECTED_OPERATORS = frozenset((
    "freq", "abs", "log", "sqrt", "square", "sigmoid", "round",
    "residual", "min", "max", "+", "-", "*", "/",
    "GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
    "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank",
    "Combine", "CombineThenFreq", "GroupByThenNUnique",
))


def supported_operators():
    actual = set(all_operators + num_operators + num_num_operators +
                 cat_num_operators + cat_cat_operators)
    if actual != EXPECTED_OPERATORS or len(actual) != 23:
        raise RuntimeError(f"unexpected OpenFE operator registry: {sorted(actual)}")
    return actual


def row_features(history, n, maximum, view):
    """Only completed draws in the most recent 5/10 are used as features."""
    last5, last10 = history[-5:], history[-10:]
    scoped = last5 if view == "latest_5" else last10
    gap = next((age for age, draw in enumerate(reversed(scoped), 1) if n in draw),
               len(scoped) + 1)
    recency = min(gap // 3, 4)
    result = {
        "number": float(n), "scaled": float(n / maximum),
        "odd": float(n % 2),
        "view_frequency": float(sum(n in d for d in scoped) / len(scoped)),
        "gap_view": float(gap),
        "repeat_last": float(n in history[-1]),
        "parity_group": str(n % 2),
        "number_band": str((n - 1) // 10),
        "recency_group": str(recency),
    }
    if view != "latest_5":
        result["freq10"] = float(sum(n in d for d in last10))
    if view != "latest_10":
        result["freq5"] = float(sum(n in d for d in last5))
    if view == "full_archive":
        result["decay10"] = float(sum((i + 1) * (n in d) for i, d in enumerate(last10)) / 55)
    return result


def feature_frame(records):
    frame = pd.DataFrame(records)
    for col in CATEGORY_COLUMNS:
        frame[col] = frame[col].astype("category")
    return frame


def anchors(total, view):
    ends = list(range(10, total))
    limit = VIEW_ANCHORS[view]
    if limit is not None:
        return ends[-limit:]
    if len(ends) > 100:
        positions = {round(i * (len(ends) - 1) / 99) for i in range(100)}
        ends = [ends[i] for i in sorted(positions)]
    return ends


def train_and_current(draws, maximum, count, view):
    x, y = [], []
    ends = anchors(len(draws), view)
    for end in ends:
        for n in range(1, maximum + 1):
            x.append(row_features(draws[:end], n, maximum, view))
            y.append(int(n in draws[end]))
    current = [row_features(draws, n, maximum, view) for n in range(1, maximum + 1)]
    train = feature_frame(x)
    test = feature_frame(current)
    if len(ends) < 5 or set(y) != {0, 1}:
        raise RuntimeError("at least five labeled historical anchors are required")
    for col in CATEGORY_COLUMNS:
        categories = sorted(set(train[col].cat.categories) | set(test[col].cat.categories))
        dtype = pd.CategoricalDtype(categories=categories)
        train[col], test[col] = train[col].astype(dtype), test[col].astype(dtype)
    return train, pd.Series(y, name="drawn"), test, ends


def audit_and_score(train, labels, current, maximum, view):
    supported = supported_operators()
    numeric = [col for col in train if col not in CATEGORY_COLUMNS]
    candidates = get_candidate_features(
        numerical_features=numeric, categorical_features=list(CATEGORY_COLUMNS), order=1
    )
    represented = {}
    for node in candidates:
        represented.setdefault(node.name, node)
    if set(represented) != supported:
        raise RuntimeError(f"operator generation incomplete: {sorted(supported - set(represented))}")
    # This is a real calculation, not an inspection of candidate names alone.
    witnesses = []
    for name in sorted(supported):
        node = represented[name]
        values = node.calculate(train)
        if len(values) != len(train) or pd.Series(values).notna().sum() == 0:
            raise RuntimeError(f"operator execution returned no valid data: {name}")
        witnesses.append(node)

    n_anchors = len(train) // maximum
    split_anchors = max(1, min(n_anchors - 1, int(n_anchors * 0.8)))
    split = split_anchors * maximum
    if split >= len(train):
        raise RuntimeError("empty chronological feature-validation interval")
    temp_name = os.path.join(tempfile.gettempdir(), f"of_lgbm_23ops_{os.getpid()}_{view}.feather")
    # OpenFE may create/remove this file. A unique name prevents cross-job collisions.
    generated = OpenFE().fit(
        data=train.copy(), label=labels.to_frame(),
        task="classification", n_jobs=1, verbose=False,
        categorical_features=list(CATEGORY_COLUMNS),
        candidate_features_list=candidates,
        train_index=list(range(split)),
        val_index=list(range(split, len(train))),
        tmp_save_path=temp_name,
        seed=20261115,
    )
    chosen, chosen_formulas = [], set()
    for node in witnesses + generated[:2]:
        formula = tree_to_formula(node)
        if formula not in chosen_formulas:
            chosen.append(node)
            chosen_formulas.add(formula)
    model_operators = {node.name for node in chosen}
    if model_operators != supported:
        raise RuntimeError(f"operator modeling incomplete: {sorted(supported - model_operators)}")
    augmented_train, augmented_current = transform(train.copy(), current.copy(), chosen, n_jobs=1)
    if len(augmented_train) != len(train) or len(augmented_current) != maximum:
        raise RuntimeError("transform lost training or candidate rows")
    for frame in (augmented_train, augmented_current):
        num_cols = frame.select_dtypes(include=[np.number]).columns
        frame[num_cols] = frame[num_cols].replace([np.inf, -np.inf], np.nan)
    model = LGBMClassifier(
        n_estimators=35, num_leaves=7, max_depth=3,
        random_state=20261115, n_jobs=1, verbosity=-1,
    )
    model.fit(augmented_train, labels)
    scores = model.predict_proba(augmented_current)[:, 1]
    if len(scores) != maximum or not np.isfinite(scores).all():
        raise RuntimeError("LightGBM produced missing or non-finite scores")
    return scores, {
        "candidate_features": len(candidates),
        "openfe_ranked_features": len(generated),
        "supported_operators": sorted(supported),
        "operator_candidates_enabled": len(represented),
        "operator_witnesses_executed": len(witnesses),
        "operator_families_in_model": len(model_operators),
        "model_feature_nodes": len(chosen),
        "chronological_train_anchors": split_anchors,
        "chronological_validation_anchors": n_anchors - split_anchors,
    }


def run(game, csv_path=None):
    maximum, count, second_max = CONFIGS[game]
    if csv_path:
        draws, second, cutoff = read_real_csv(csv_path, game)
    else:
        rng = random.Random(20261115 + maximum)
        draws = [tuple(sorted(rng.sample(range(1, maximum + 1), count))) for _ in range(18)]
        second, cutoff = [], None
    if len(draws) < 16:
        raise RuntimeError("at least 16 completed draws needed")
    results, audits = {}, {}
    for view, weight in VIEW_WEIGHTS.items():
        train, labels, current, ends = train_and_current(draws, maximum, count, view)
        score, audit = audit_and_score(train, labels, current, maximum, view)
        results[view] = score
        audits[view] = {**audit, "anchor_count": len(ends), "weight": weight}
    fused = sum(VIEW_WEIGHTS[k] * results[k] for k in VIEW_WEIGHTS)
    if not np.isfinite(fused).all() or not math.isclose(sum(VIEW_WEIGHTS.values()), 1.0):
        raise RuntimeError("invalid three-view fusion")
    pool = sorted(range(1, maximum + 1), key=lambda n: (-fused[n - 1], n))[:8]
    tickets = list(combinations(pool, count))
    ticket = max(tickets, key=lambda t: (sum(float(fused[n - 1]) for n in t), tuple(-n for n in t)))
    second_pick = None
    if second_max:
        second_pick = max(
            range(1, second_max + 1),
            key=lambda n: (0.6 * sum(x == n for x in second[-5:]) +
                           0.4 * sum(x == n for x in second[-10:]), -n)
        ) if csv_path else 1
    output = {
        "system": "independent OF-LGBM 23-operator research E2E",
        "game": game, "first_zone": sorted(ticket), "second_zone": second_pick,
        "three_view_weights": VIEW_WEIGHTS, "per_view_audits": audits,
        "openfe_23_operator_candidate_enablement": True,
        "openfe_23_operator_execution_verified": True,
        "all_23_operator_families_passed_to_lightgbm": True,
        "lightgbm_trained_and_scored": True,
        "exact_candidates": len(tickets),
        "history_draws": len(draws), "data_cutoff": cutoff,
        "history_source": "repository CSV; official external verification pending" if csv_path else "deterministic synthetic fixture",
        "synthetic_data": not bool(csv_path),
        "official_data_verified": False,
        "out_of_sample_predictive_advantage_verified": False,
        "second_zone_model": "recent 5/10 frequency baseline; not OpenFE/LightGBM" if second_max else None,
    }
    if csv_path:
        import hashlib
        output["data_sha256"] = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    assert len(output["first_zone"]) == count
    assert len(set(output["first_zone"])) == count
    assert all(1 <= n <= maximum for n in output["first_zone"])
    assert all(a["operator_families_in_model"] == 23 for a in audits.values())
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=tuple(CONFIGS), default="power")
    p.add_argument("--csv", help="chronological verified-format historical lottery CSV")
    p.add_argument("--output", help="write JSON evidence to disk")
    args = p.parse_args()
    result = run(args.game, args.csv)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
