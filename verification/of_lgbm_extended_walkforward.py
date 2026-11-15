#!/usr/bin/env python3
"""Standalone OF-LGBM retrospective multi-horizon prequential evaluation.

All predictions use actual OpenFE + LightGBM refits via predict_one.
No V6, CUDA-Q, QAOA, QUBO, or external prediction services.
Full official draw-by-draw provenance and prospective predictive edge are unverified.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from of_lgbm_all_operator_e2e import CONFIGS, VIEW_WEIGHTS
from of_lgbm_forward_audit import audit_history, predict_one, theory_3plus
from of_lgbm_real_history_gate import audit_csv

MIN_HISTORY = 25
HORIZONS = (100, 200, 500)


def null_tail(k, n, p):
    """One-sided fair-null probability, unadjusted for repeated comparisons."""
    return math.fsum(
        math.comb(n, i) * p ** i * (1 - p) ** (n - i)
        for i in range(k, n + 1)
    )


def summarize(outcomes, game, horizons=HORIZONS):
    maximum, count, _ = CONFIGS[game]
    theory = theory_3plus(maximum, count)
    report = {}
    for horizon in horizons:
        if horizon > len(outcomes):
            report[str(horizon)] = {
                "status": "not_completed",
                "required_folds": horizon,
                "executed_folds": len(outcomes),
            }
            continue
        tail = outcomes[-horizon:]
        hits = sum(row["matches"] >= 3 for row in tail)
        report[str(horizon)] = {
            "status": "completed",
            "folds": horizon,
            "target_start_id": tail[0]["target_id"],
            "target_end_id": tail[-1]["target_id"],
            "hit_distribution": {
                str(i): sum(row["matches"] == i for row in tail)
                for i in range(count + 1)
            },
            "hits_3plus": hits,
            "rate_3plus": hits / horizon,
            "fair_null_rate_3plus": theory,
            "single_random_baseline_hits_3plus": sum(
                row["random_matches"] >= 3 for row in tail
            ),
            "unadjusted_null_tail_p_value": null_tail(hits, horizon, theory),
        }
    return report


def run(game, csv_file, horizons=HORIZONS, max_folds=None, checkpoint=None):
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must all be positive")
    gate = audit_csv(csv_file, game)
    ids, draws, seconds, archive = audit_history(csv_file, game)
    if gate["rows"] != archive["rows"] or gate["last_id"] != ids[-1].__str__():
        raise RuntimeError("source audits disagree")
    maximum, count, second_max = CONFIGS[game]
    available = len(draws) - MIN_HISTORY
    feasible = [h for h in horizons if h <= available]
    target_folds = max(feasible, default=0)
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be positive")
        target_folds = min(target_folds or available, max_folds)
    if target_folds < 1:
        raise ValueError("insufficient history even for one holdout")
    outcomes = []
    start = len(draws) - target_folds
    for i in range(start, len(draws)):
        ticket, picked_second = predict_one(
            draws[:i], seconds[:i] if second_max else [], game
        )
        actual = draws[i]
        baseline = random.Random(20261115 + i).sample(
            range(1, maximum + 1), count
        )
        outcomes.append({
            "target_id": str(ids[i]),
            "cutoff_id": str(ids[i - 1]),
            "ticket": list(ticket),
            "actual": list(actual),
            "matches": len(set(ticket) & set(actual)),
            "random_matches": len(set(baseline) & set(actual)),
            "second_pick": picked_second,
            "second_actual": seconds[i] if second_max else None,
        })
        if checkpoint and (len(outcomes) % 10 == 0 or i == len(draws) - 1):
            Path(checkpoint).write_text(json.dumps({
                "game": game, "executed_folds": len(outcomes),
                "target_folds": target_folds, "outcomes": outcomes,
                "partial": len(outcomes) < target_folds,
            }, indent=2) + "\n", encoding="utf-8")
    if any(row["target_id"] == row["cutoff_id"] for row in outcomes):
        raise AssertionError("future leakage: target and cutoff are identical")
    return {
        "system": "standalone OF-LGBM extended causal walk-forward",
        "game": game,
        "csv_rows": len(draws), "csv_cutoff_id": str(ids[-1]),
        "csv_sha256": gate["sha256"],
        "csv_official_provenance_verified": False,
        "forward_looking_blind_test": False,
        "predictive_advantage_verified": False,
        "view_weights": VIEW_WEIGHTS,
        "feature_window_lengths": [10, 20],
        "full_archive_training": "all eligible prior anchors",
        "actual_openfe_lightgbm_refit_each_fold": True,
        "maximum_possible_folds_with_current_csv": available,
        "executed_folds": len(outcomes),
        "horizons": summarize(outcomes, game, horizons),
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--max-folds", type=int,
                        help="smoke/debug cap; does not count as full validation")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.game, args.csv, tuple(args.horizons),
                 args.max_folds, args.checkpoint)
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    print(json.dumps({
        "game": result["game"], "executed_folds": result["executed_folds"],
        "horizons": result["horizons"],
        "official_provenance_verified": False,
        "predictive_advantage_verified": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
