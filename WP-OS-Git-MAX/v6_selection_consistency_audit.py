#!/usr/bin/env python3
"""Fail-closed independent final selection audit for V6 three-view output."""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

def audit(result):
    layer = result["openfe_feature_layer"]
    if layer.get("mode") != "three_history_views_100_percent":
        raise ValueError("three-view OpenFE mode not enabled")
    if layer.get("full_archive_role") != "training_anchors_only":
        raise ValueError("unexpected full-archive semantics")
    views = ("latest_5", "latest_10", "full_archive")
    weights = layer["view_weights"]
    if set(weights) != set(views) or any(not math.isclose(weights[k], 1/3, rel_tol=0, abs_tol=1e-12) for k in views):
        raise ValueError("incorrect view weights")
    by_view = layer["per_view_scores"]
    scores = {}
    for n in range(1, 39):
        vals = []
        for view in views:
            v = by_view[view]
            x = v.get(str(n), v.get(n))
            if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError("missing or invalid per-view score")
            vals.append(x)
        scores[n] = sum(vals)/3
    pool = sorted(scores, key=lambda n: (-scores[n], n))[:8]
    candidates = list(combinations(sorted(pool), 6))
    expected = max(candidates, key=lambda c: (sum(scores[n] for n in c), tuple(-n for n in c)))
    opt = result["exact_optimization_layer"]
    selected = result["first_zone"]
    if len(selected) != 6 or len(set(selected)) != 6 or any(n not in pool for n in selected):
        raise ValueError("invalid final first-zone selection")
    if tuple(selected) != expected or selected != opt["exact_selection"]:
        raise ValueError("final selection differs from independent exhaustive optimum")
    if opt["searched_combinations"] != 28 or opt["globally_optimal_within_pool"] is not True:
        raise ValueError("exact-search evidence inconsistent")
    expected_score = sum(scores[n] for n in expected)
    if not math.isclose(opt["exact_score"], expected_score, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("exact-search score inconsistent")
    approx = opt["approximate_selection"]
    if len(approx) != 6 or len(set(approx)) != 6 or not set(approx).issubset(pool):
        raise ValueError("invalid CUDA-Q diagnostic selection")
    approx_score = sum(scores[n] for n in approx)
    if not math.isclose(opt["approximate_score"], approx_score, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("approximate score inconsistent")
    if not math.isclose(opt["score_gap"], max(0, expected_score-approx_score), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("diagnostic/exact score gap inconsistent")
    if not isinstance(result["second_zone"], int) or not 1 <= result["second_zone"] <= 8:
        raise ValueError("invalid second-zone selection")
    if result["views"]["full_archive"]["rows"] < 12:
        raise ValueError("insufficient input archive")
    return {"status": "PASS", "scope": "final selection arithmetic and mapping only",
            "first_zone": selected, "second_zone": result["second_zone"],
            "candidate_pool": sorted(pool), "checked_combinations": 28,
            "score_gap": opt["score_gap"],
            "full_archive_is_officially_complete": False,
            "predictive_advantage_verified": False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(json.loads(Path(args.result).read_text(encoding="utf-8")))
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(data, encoding="utf-8")
    print(data)

if __name__ == "__main__":
    main()
