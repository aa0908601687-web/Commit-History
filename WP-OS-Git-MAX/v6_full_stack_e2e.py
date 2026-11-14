#!/usr/bin/env python3
"""Experimental single-run V6 OF3Q integrated CPU E2E (NOT production).

Runs real OpenFE + LightGBM and tsfresh + Featuretools candidate scoring,
then local CUDA-Q qpp-cpu, CPU Ising/QUBO, OR-Tools CP-SAT, and exact
8-choose-6 reference under one invocation. This is not a predictive claim.
Uses only local CPU simulation; does not claim live QPU execution or external inference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def normalize(scores):
    if set(scores) != set(range(1, 39)) or not all(math.isfinite(float(v)) for v in scores.values()):
        raise ValueError("expected finite candidate scores for exactly first-zone numbers 1..38")
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {n: 0.0 for n in range(1, 39)}
    return {n: (float(scores[n]) - float(lo)) / (float(hi) - float(lo)) for n in range(1, 39)}


def fuse_scores(openfe, advanced, openfe_weight=0.5):
    if not 0 <= openfe_weight <= 1:
        raise ValueError("openfe_weight outside [0,1]")
    a, b = normalize(openfe), normalize(advanced)
    return {n: openfe_weight * a[n] + (1 - openfe_weight) * b[n] for n in range(1, 39)}


def run(csv_path, openfe_weight=0.5):
    from v6_exact_e2e import complete_result, validate
    from v6_openfe_bridge import score_three_history_views
    from v6_advanced_feature_stack import build_features

    rows = validate(csv_path)
    openfe_scores, openfe_meta = score_three_history_views(rows, n_jobs=1)
    advanced_scores, advanced_meta = build_features(rows)
    scores = fuse_scores(openfe_scores, advanced_scores, openfe_weight)
    feature_layer = {
        "enabled": True,
        "weight": 1.0,
        "engine": "OpenFE+LightGBM AND tsfresh+Featuretools",
        "feature_window": [5, 10],
        "openfe_weight": openfe_weight,
        "advanced_weight": 1 - openfe_weight,
        "openfe": openfe_meta,
        "advanced": advanced_meta,
        "fused_scores": {str(n): scores[n] for n in range(1, 39)},
    }
    result = complete_result(
        csv_path, rows, scores, feature_layer,
        use_cudaq=True, require_cudaq=True,
        ising_qubo_check=True, ortools_check=True
    )
    assert result["cudaq_layer"]["executed"] is True
    assert result["ising_qubo_layer"]["executed"] is True
    assert result["ortools_layer"]["executed"] is True
    assert result["exact_optimization_layer"]["searched_combinations"] == 28
    result["full_stack_integration"] = {
        "executed": True,
        "feature_engines": ["OpenFE", "LightGBM", "tsfresh", "Featuretools"],
        "optimization_engines": ["CUDA-Q qpp-cpu", "CPU Ising/QUBO", "OR-Tools CP-SAT", "Python exact"],
        "production_replacement": False,
        "predictive_advantage_verified": False,
        "real_qpu": False,
    }
    result["model"] = "V6 OF3Q combined experimental real feature and CPU optimizer E2E"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--openfe-weight", type=float, default=0.5)
    args = p.parse_args()
    result = run(args.csv, args.openfe_weight)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"last_draw": result["last_draw"], "first_zone": result["first_zone"],
                      "second_zone": result["second_zone"], "full_stack_integration": result["full_stack_integration"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
