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

import csv
import hashlib
from of_lgbm_ticket_selection import select_ticket
from of_lgbm_real_history_gate import audit_csv

CONFIGS = {"power": (38, 6, 8), "lotto": (49, 6, None), "539": (39, 5, None)}


def read_real_csv(csv_path, kind):
    """Strict chronological and range validation, independent of QUBO/CUDA-Q."""
    maximum, count, second_max = CONFIGS[kind]
    # Enforce exact sequence, year rollover, ranges and dates when supplied.
    audit_csv(csv_path, kind, min_rows=25)
    rows, second_rows = [], []
    previous = -1
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"draw_id", "first"} | ({"second"} if second_max else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"required CSV columns: {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            try:
                draw_id = int(row["draw_id"])
                first = tuple(sorted(int(v) for v in row["first"].replace(",", " ").split()))
                second = int(row["second"]) if second_max else None
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid CSV on line {line}") from exc
            if draw_id <= previous:
                raise ValueError(f"non-increasing draw ID on line {line}")
            if len(first) != count or len(set(first)) != count or not all(1 <= n <= maximum for n in first):
                raise ValueError(f"invalid first-zone numbers on line {line}")
            if second_max and not (1 <= second <= second_max):
                raise ValueError(f"invalid second-zone number on line {line}")
            rows.append(first)
            if second_max:
                second_rows.append(second)
            previous = draw_id
    if len(rows) < 25:
        raise ValueError("at least 25 completed draws are required")
    return rows, second_rows, str(previous)

VIEW_ANCHORS = {"latest_10": 10, "latest_20": 20, "full_archive": None}
# No fixed three-view score-fusion coefficients. Each forecast uses earlier draws only.
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
    """Use only prior completed draws: latest 10, latest 20, or complete history."""
    if view not in VIEW_ANCHORS:
        raise ValueError(f"unknown OF-LGBM view: {view}")
    last10, last20 = history[-10:], history[-20:]
    scoped = {"latest_10": last10, "latest_20": last20, "full_archive": history}[view]
    gap = next((age for age, draw in enumerate(reversed(scoped), 1) if n in draw),
               len(scoped) + 1)
    recency = min(gap // 3, 4)
    hits = sum(n in d for d in scoped)
    result = {
        "number": float(n), "scaled": float(n / maximum),
        "odd": float(n % 2),
        "view_frequency": float(hits / len(scoped)),
        "gap_view": float(gap),
        "repeat_last": float(n in history[-1]),
        "parity_group": str(n % 2),
        "number_band": str((n - 1) // 10),
        "recency_group": str(recency),
    }
    if view != "latest_10":
        result["freq20"] = float(sum(n in d for d in last20))
    if view != "latest_20":
        result["freq10"] = float(sum(n in d for d in last10))
    if view == "full_archive":
        result["historical_count"] = float(hits)
        result["historical_draws"] = float(len(scoped))
        result["decay20"] = float(
            sum((i + 1) * (n in d) for i, d in enumerate(last20)) / 210
        )
    return result


def causal_view_weights(draws, maximum, calibration_draws=30):
    """Fit view fusion from past-only one-step frequency baselines (research heuristic).

    This calibration does NOT evaluate LightGBM itself and is not a validated
    optimal ML ensemble. All targets precede the draw being forecast.
    """
    names = tuple(VIEW_ANCHORS)
    starts = range(max(20, len(draws) - calibration_draws), len(draws))
    errors = {name: 0.0 for name in names}
    used = 0
    for end in starts:
        prior = draws[:end]
        target = set(draws[end])
        for name in names:
            limit = VIEW_ANCHORS[name]
            history = prior[-limit:] if limit is not None else prior
            counts = {n: sum(n in row for row in history) for n in range(1, maximum + 1)}
            # Laplace smoothing and a proper Brier loss on the next historical draw.
            errors[name] += sum(((counts[n] + 1) / (len(history) + 2) - int(n in target)) ** 2
                                for n in range(1, maximum + 1)) / maximum
        used += 1
    if not used:
        return {name: 1.0 / len(names) for name in names}
    losses = {name: errors[name] / used for name in names}
    best = min(losses.values())
    strengths = {name: math.exp(-30.0 * (losses[name] - best)) for name in names}
    total = sum(strengths.values())
    return {name: strengths[name] / total for name in names}


def feature_frame(records):
    frame = pd.DataFrame(records)
    for col in CATEGORY_COLUMNS:
        frame[col] = frame[col].astype("category")
    return frame


def anchors(total, view):
    # 20 completed draws warm up the 20-draw feature window.
    # All eligible prior historical training anchors are used (no 100-anchor cap).
    ends = list(range(20, total))
    limit = VIEW_ANCHORS[view]
    return ends[-limit:] if limit is not None else ends


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
    # Bound actual OpenFE feature screening while keeping all 23 operator kinds.
    # Every operator has at least one real candidate passed to OpenFE.fit().
    screen_candidates = []
    seen_counts = {}
    for node in candidates:
        if seen_counts.get(node.name, 0) < 2:
            screen_candidates.append(node)
            seen_counts[node.name] = seen_counts.get(node.name, 0) + 1
    assert set(seen_counts) == supported
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
        candidate_features_list=screen_candidates,
        train_index=pd.Index(range(split)),
        val_index=pd.Index(range(split, len(train))),
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
        "candidate_features_possible": len(candidates),
        "candidate_features_screened": len(screen_candidates),
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
        draws = [tuple(sorted(rng.sample(range(1, maximum + 1), count))) for _ in range(30)]
        second, cutoff = [], None
    if len(draws) < 25:
        raise RuntimeError("at least 16 completed draws needed")
    results, audits = {}, {}
    view_weights = causal_view_weights(draws, maximum)
    for view, weight in view_weights.items():
        train, labels, current, ends = train_and_current(draws, maximum, count, view)
        score, audit = audit_and_score(train, labels, current, maximum, view)
        results[view] = score
        audits[view] = {**audit, "anchor_count": len(ends), "weight": weight}
    fused = sum(view_weights[k] * results[k] for k in view_weights)
    if not np.isfinite(fused).all() or not math.isclose(sum(view_weights.values()), 1.0):
        raise RuntimeError("invalid three-view fusion")
    ticket, selection_audit = select_ticket(fused, draws, count, maximum)
    second_pick = None
    if second_max:
        second_pick = max(
            range(1, second_max + 1),
            key=lambda n: (0.6 * sum(x == n for x in second[-10:]) +
                           0.4 * sum(x == n for x in second[-20:]), -n)
        ) if csv_path else 1
    output = {
        "system": "independent OF-LGBM 23-operator research E2E",
        "game": game, "first_zone": sorted(ticket), "second_zone": second_pick,
        "three_view_weights": view_weights, "view_weight_policy": "causal_frequency_brier_calibration_not_ML", "per_view_audits": audits,
        "openfe_23_operator_candidate_enablement": True,
        "openfe_23_operator_execution_verified": True,
        "all_23_operator_families_passed_to_lightgbm": True,
        "lightgbm_trained_and_scored": True,
        "exact_candidates": selection_audit["candidate_combinations"],
        "selection_audit": selection_audit,
        "history_draws": len(draws), "data_cutoff": cutoff,
        "feature_window_lengths": [10, 20], "full_archive_training_anchors": len(draws) - 20,
        "history_source": "repository CSV; official external verification pending" if csv_path else "deterministic synthetic fixture",
        "synthetic_data": not bool(csv_path),
        "official_data_verified": False,
        "out_of_sample_predictive_advantage_verified": False,
        "second_zone_model": "recent 10/20 frequency baseline; not OpenFE/LightGBM" if second_max else None,
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
