#!/usr/bin/env python3
"""Leakage-aware, prequential diagnostics for V6 first-zone scores.

Research diagnostics only: scores are not certified probabilities, and
neither a passing test nor an improved historical metric implies lottery edge.
"""
from __future__ import annotations
import argparse
import json
import math
from statistics import mean

FIRST_ZONE = tuple(range(1, 39))
BASE_RATE = 6 / 38


def validate_scores(scores):
    if set(scores) != set(FIRST_ZONE):
        raise ValueError("expected exactly one score for every first-zone number 1..38")
    if any(not isinstance(scores[n], (int, float)) or not math.isfinite(scores[n]) for n in FIRST_ZONE):
        raise ValueError("feature scores must be finite real numbers")
    if any(not 0 <= scores[n] <= 1 for n in FIRST_ZONE):
        raise ValueError("LightGBM scores must be in [0,1]")
    return {n: float(scores[n]) for n in FIRST_ZONE}


def top_numbers(scores, size):
    return sorted(sorted(FIRST_ZONE, key=lambda n: (-scores[n], n))[:size])


def brier(scores, winning):
    wins = set(winning)
    return mean((scores[n] - int(n in wins)) ** 2 for n in FIRST_ZONE)


def one_step(history, target, scorer):
    """Only completed past draws enter scorer; target is used strictly for evaluation."""
    raw, metadata = scorer(history)
    scores = validate_scores(raw)
    winning = set(target[1])
    selected = top_numbers(scores, 6)
    pool = top_numbers(scores, 8)
    return {
        "train_through": history[-1][0],
        "target_draw_id": target[0],
        "selected": selected,
        "candidate_pool": pool,
        "hits": len(winning.intersection(selected)),
        "brier": brier(scores, winning),
        "baseline_brier": BASE_RATE * (1 - BASE_RATE),
        "generated_features": metadata.get("generated_features"),
        "used_features": metadata.get("used_features"),
    }


def walk_forward(rows, scorer, min_history=12, max_steps=None):
    if min_history < 12:
        raise ValueError("OpenFE requires at least 12 completed draws")
    if len(rows) <= min_history:
        raise ValueError("not enough draws for an out-of-sample evaluation")
    ends = list(range(min_history, len(rows)))
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        ends = ends[-max_steps:]
    results = [one_step(rows[:end], rows[end], scorer) for end in ends]
    return {
        "method": "prequential expanding-window; target draw excluded from fitting",
        "selection_feature_windows": [5, 10],
        "evaluated_draws": len(results),
        "target_range": [results[0]["target_draw_id"], results[-1]["target_draw_id"]],
        "mean_brier": mean(r["brier"] for r in results),
        "fair_baseline_brier": BASE_RATE * (1 - BASE_RATE),
        "mean_hits": mean(r["hits"] for r in results),
        "three_plus_count": sum(r["hits"] >= 3 for r in results),
        "per_draw": results,
        "caveat": "Exploratory performance; model features/parameters must be fixed before an untouched evaluation.",
    }


def leave_one_recent_draw_out(rows, scorer):
    """Sensitivity diagnostic only, NOT a valid future draw simulation.

    Drop each of the latest ten draws, refit and compare first-zone rankings.
    A changed ten-draw window is expected and must not be called an error.
    """
    if len(rows) < 13:
        raise ValueError("at least 13 draws required for sensitivity audit")
    original = validate_scores(scorer(rows)[0])
    reference_pool = set(top_numbers(original, 8))
    reference_selected = set(top_numbers(original, 6))
    trials = []
    for idx in range(len(rows) - 10, len(rows)):
        altered = rows[:idx] + rows[idx + 1:]
        candidate = validate_scores(scorer(altered)[0])
        selection = set(top_numbers(candidate, 6))
        pool = set(top_numbers(candidate, 8))
        trials.append({
            "omitted_draw": rows[idx][0],
            "top6_overlap": len(selection & reference_selected),
            "pool8_overlap": len(pool & reference_pool),
            "max_abs_score_change": max(abs(candidate[n] - original[n]) for n in FIRST_ZONE),
        })
    return {
        "scope": "leave-one-recent-draw-out refit sensitivity; not predictive error",
        "reference_last_draw": rows[-1][0],
        "baseline_selection": sorted(reference_selected),
        "mean_top6_overlap": mean(t["top6_overlap"] for t in trials),
        "mean_pool8_overlap": mean(t["pool8_overlap"] for t in trials),
        "trials": trials,
    }


def main():
    from v6_ai_exact_e2e import validate
    from v6_openfe_bridge import score_three_history_views
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    rows = validate(args.csv)
    scorer = lambda history: score_three_history_views(history, n_jobs=1)
    report = {
        "walk_forward": walk_forward(rows, scorer, max_steps=args.max_steps),
        "sensitivity": None if args.skip_sensitivity else leave_one_recent_draw_out(rows, scorer),
        "source_last_draw": rows[-1][0],
        "model": "PR43 OpenFE three views/LightGBM",
        "predictive_advantage_verified": False,
    }
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    print(output)


if __name__ == "__main__":
    main()
