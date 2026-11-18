#!/usr/bin/env python3
"""Optional 100/200/500-draw genuine ML walk-forward; never an accuracy guarantee.

High-cost CPU workload; distinct from mandatory five-fold CI. Every forecast is
produced with preceding completed draws only by the existing OF-LGBM runner.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from of_lgbm_forward_audit import audit_history, walk_forward, CONFIGS, theory_3plus

ALLOWED_FOLDS = (100, 200, 500)

def wilson_interval(hits, n, z=1.959963984540054):
    if n <= 0 or not 0 <= hits <= n:
        raise ValueError("invalid proportion")
    p = hits / n
    den = 1 + z*z/n
    middle = (p+z*z/(2*n))/den
    margin = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0., middle-margin), min(1., middle+margin)]

def make_summary(ids, draws, seconds, game, folds, runner=walk_forward):
    if folds not in ALLOWED_FOLDS:
        raise ValueError("folds must be 100, 200, or 500")
    if len(ids) != len(draws) or (game == "power" and len(seconds) != len(draws)):
        raise ValueError("misaligned input history")
    if len(draws)-folds < 25:
        raise ValueError("at least 25 earlier draws required before first holdout")
    result = runner(ids, draws, seconds, game, folds)
    outcomes = result["outcomes"]
    if len(outcomes) != folds:
        raise ValueError("incomplete forecast outcomes")
    expected_targets = [str(x) for x in ids[-folds:]]
    if [x["target_id"] for x in outcomes] != expected_targets:
        raise ValueError("holdout target order mismatch")
    for i, row in enumerate(outcomes):
        if row["cutoff_id"] != str(ids[len(ids)-folds+i-1]):
            raise ValueError("future-data leakage or wrong cutoff")
    maximum, count, _ = CONFIGS[game]
    distribution = {str(k): sum(o["matches"] == k for o in outcomes) for k in range(count+1)}
    random_distribution = {str(k): sum(o["random_matches"] == k for o in outcomes) for k in range(count+1)}
    hits = sum(o["matches"] >= 3 for o in outcomes)
    baseline_hits = sum(o["random_matches"] >= 3 for o in outcomes)
    return {
      "game": game, "folds": folds,
      "model_0_to_n_matches": distribution,
      "sampled_random_0_to_n_matches": random_distribution,
      "model_3plus_hits": hits,
      "model_3plus_rate": hits / folds,
      "model_3plus_wilson_95pct": wilson_interval(hits, folds),
      "single_seed_random_3plus_hits": baseline_hits,
      "single_seed_random_3plus_rate": baseline_hits / folds,
      "fair_exact_3plus_rate": theory_3plus(maximum, count),
      "fair_expected_3plus_count": folds * theory_3plus(maximum, count),
      "prediction_error_audit": result["prediction_error_audit"],
      "full_walk_forward": result,
      "official_sources_row_verified": False,
      "predictive_advantage_verified": False,
      "out_of_sample_type": "historical prequential exploratory, not untouched prospective blind test",
      "note": "Single-seed random is illustrative; descriptive confidence interval does not adjust for model selection, repeated testing, or multiple comparisons.",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--folds", type=int, choices=ALLOWED_FOLDS, default=100)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    ids, draws, seconds, source_audit = audit_history(a.csv, a.game)
    result = make_summary(ids, draws, seconds, a.game, a.folds)
    result["source_audit"] = source_audit
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "game": a.game, "folds": a.folds,
        "model_3plus_hits": result["model_3plus_hits"],
        "fair_exact_3plus_rate": result["fair_exact_3plus_rate"],
        "official_sources_row_verified": False,
        "predictive_advantage_verified": False,
        "output": a.output,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
