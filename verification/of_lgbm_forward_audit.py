#!/usr/bin/env python3
"""Independent OF-LGBM historical CSV audit and strict prequential ML holdout.

Each test draw is predicted using only earlier completed draws, with actual
OpenFE+LightGBM fitting in all three views. No V6, QAOA, CUDA-Q, or QUBO imports.
Official-source attestation and prediction advantage are NOT asserted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from of_lgbm_all_operator_e2e import (
    CONFIGS, VIEW_WEIGHTS, train_and_current, audit_and_score,
)


def audit_history(path, game):
    maximum, count, second_max = CONFIGS[game]
    path = Path(path)
    ids, draws, seconds = [], [], []
    years = Counter()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"draw_id", "first"} | ({"second"} if second_max else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("missing required CSV columns")
        prev_year = prev_seq = None
        for line, row in enumerate(reader, 2):
            try:
                draw_id = int(row["draw_id"])
                first = tuple(sorted(int(x) for x in row["first"].replace(",", " ").split()))
                second = int(row["second"]) if second_max else None
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid CSV at line {line}") from exc
            year, seq = divmod(draw_id, 1_000_000)
            if year < 1 or seq < 1:
                raise ValueError(f"invalid draw id at line {line}")
            if prev_year is not None:
                if year == prev_year and seq != prev_seq + 1:
                    raise ValueError(f"nonconsecutive draw at line {line}")
                if year != prev_year and (year != prev_year + 1 or seq != 1):
                    raise ValueError(f"invalid year transition at line {line}")
            if len(first) != count or len(set(first)) != count or any(n < 1 or n > maximum for n in first):
                raise ValueError(f"invalid first zone at line {line}")
            if second_max and not 1 <= second <= second_max:
                raise ValueError(f"invalid second zone at line {line}")
            ids.append(draw_id)
            draws.append(first)
            if second_max:
                seconds.append(second)
            years[year] += 1
            prev_year, prev_seq = year, seq
    if len(draws) < 17:
        raise ValueError("at least 17 completed draws required")
    summary = {
        "rows": len(draws), "first_id": str(ids[0]), "last_id": str(ids[-1]),
        "years": dict(sorted(years.items())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chronological_and_legal_range_verified": True,
        "independent_official_provenance_verified": False,
        "dates_verified": False,
    }
    return ids, draws, seconds, summary


def predict_one(prior_draws, prior_seconds, game):
    maximum, count, second_max = CONFIGS[game]
    if len(prior_draws) < 25:
        raise ValueError("at least 25 earlier draws required")
    per_view = []
    for view, weight in VIEW_WEIGHTS.items():
        training, labels, current, _ = train_and_current(prior_draws, maximum, count, view)
        scores, audit = audit_and_score(training, labels, current, maximum, view)
        if audit["operator_families_in_model"] != 23:
            raise RuntimeError("OpenFE family coverage incomplete")
        per_view.append(weight * scores)
    scores = np.sum(per_view, axis=0)
    if not np.isfinite(scores).all():
        raise RuntimeError("invalid LightGBM output")
    pool = sorted(range(1, maximum + 1), key=lambda n: (-scores[n - 1], n))[:8]
    ticket = max(combinations(pool, count),
                 key=lambda t: (sum(float(scores[n-1]) for n in t), tuple(-n for n in t)))
    second = None
    if second_max:
        second = max(range(1, second_max + 1),
                     key=lambda n: (0.6 * sum(x == n for x in prior_seconds[-10:]) +
                                    0.4 * sum(x == n for x in prior_seconds[-20:]), -n))
    return tuple(sorted(ticket)), second


def theory_3plus(maximum, count):
    total = math.comb(maximum, count)
    return sum(math.comb(count, h) * math.comb(maximum-count, count-h) / total
               for h in range(3, count+1))


def walk_forward(ids, draws, seconds, game, folds):
    maximum, count, second_max = CONFIGS[game]
    if folds < 1 or len(draws) - folds < 25:
        raise ValueError("invalid holdout length")
    outcomes = []
    for target in range(len(draws)-folds, len(draws)):
        before = draws[:target]
        ticket, pick2 = predict_one(before, seconds[:target] if second_max else [], game)
        actual = draws[target]
        matched = len(set(ticket) & set(actual))
        random_ticket = random.Random(20261115 + target).sample(range(1, maximum+1), count)
        outcomes.append({
            "target_id": str(ids[target]), "cutoff_id": str(ids[target-1]),
            "ticket": list(ticket), "actual": list(actual), "matches": matched,
            "random_matches": len(set(random_ticket) & set(actual)),
            "second_pick": pick2, "second_actual": seconds[target] if second_max else None,
        })
    distribution = {str(k): sum(row["matches"] == k for row in outcomes)
                    for k in range(count+1)}
    wins = sum(row["matches"] >= 3 for row in outcomes)
    return {
        "game": game, "mode": "actual OpenFE+LightGBM refit for each held-out draw",
        "folds": folds, "outcomes": outcomes, "distribution": distribution,
        "hits_3plus": wins, "rate_3plus": wins / folds,
        "fair_null_rate_3plus": theory_3plus(maximum, count),
        "random_baseline_3plus": sum(row["random_matches"] >= 3 for row in outcomes),
        "feature_window_lengths": [10, 20], "view_weights": VIEW_WEIGHTS,
        "full_archive_anchor_policy": "all_eligible_completed_draws",
        "prediction_prior_to_each_target": True,
        "official_data_verified": False,
        "predictive_advantage_verified": False,
        "note": "Exploratory historical walk-forward; not an untouched future blind test.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=tuple(CONFIGS), default="power")
    p.add_argument("--csv", required=True)
    p.add_argument("--folds", type=int, default=0,
                   help="0=audit only; 1=one true ML holdout; use 100/200/500 separately for long runs")
    p.add_argument("--output")
    args = p.parse_args()
    ids, draws, seconds, audit = audit_history(args.csv, args.game)
    result = {"audit": audit, "walk_forward": None}
    if args.folds:
        result["walk_forward"] = walk_forward(ids, draws, seconds, args.game, args.folds)
    data = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(data + "\n", encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
