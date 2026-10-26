#!/usr/bin/env python3
"""Equal-weight fusion of the WP-OS Git MAX 5/10 and 10/20 models."""
import argparse
import hashlib
import json
from collections import Counter

from wp_os_git_max import load, rank_draws as rank_5_10, score as score_5_10
from wp_os_git_max_10_20 import rank_draws as rank_10_20, rank_second as second_10_20

MODEL_VERSION = "wp-os-git-max-fusion-5-10-20-v1"
MAIN_WEIGHT = 0.5
CONTROL_WEIGHT = 0.5


def _combine(main_ranking, control_ranking):
    main = dict(main_ranking)
    control = dict(control_ranking)
    numbers = sorted(set(main) | set(control))
    ranked = [
        (number, round(MAIN_WEIGHT * main[number] + CONTROL_WEIGHT * control[number], 12))
        for number in numbers
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def predict(draws):
    if len(draws) < 20:
        raise ValueError("At least 20 completed draws are required")
    first_draws = [draw["first"] for draw in draws]
    second_values = [draw["second"] for draw in draws]
    main_first = rank_5_10(first_draws, 38)
    control_first = rank_10_20(first_draws, 38)
    main_second = score_5_10(second_values, 8)
    control_second = second_10_20(second_values, 8)
    fused_first = _combine(main_first, control_first)
    fused_second = _combine(main_second, control_second)
    payload = {
        "model_version": MODEL_VERSION,
        "through_draw": draws[-1]["draw_id"],
        "feature_windows": [5, 10, 20],
        "fusion_weights": {"main_5_10": MAIN_WEIGHT, "control_10_20": CONTROL_WEIGHT},
        "first": sorted(number for number, _ in fused_first[:6]),
        "second": fused_second[0][0],
        "top_first_scores": fused_first[:10],
        "second_scores": fused_second,
        "component_predictions": {
            "main_5_10": {
                "first": sorted(number for number, _ in main_first[:6]),
                "second": main_second[0][0],
            },
            "control_10_20": {
                "first": sorted(number for number, _ in control_first[:6]),
                "second": control_second[0][0],
            },
        },
        "method": "Equal-weight score fusion (5/10 and 10/20); no predictive claim",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return payload


def backtest(draws):
    hits = Counter()
    second_hits = 0
    for end in range(20, len(draws)):
        prediction = predict(draws[:end])
        actual = draws[end]
        matched = len(set(prediction["first"]) & set(actual["first"]))
        hits[matched] += 1
        second_hits += prediction["second"] == actual["second"]
    tested = max(0, len(draws) - 20)
    rate = lambda value: value / tested if tested else 0.0
    return {
        "model_version": MODEL_VERSION,
        "tested": tested,
        "feature_windows": [5, 10, 20],
        "fusion_weights": {"main_5_10": MAIN_WEIGHT, "control_10_20": CONTROL_WEIGHT},
        "first_zone_hits": {str(n): hits[n] for n in range(7)},
        "first_zone_3plus_rate": rate(sum(hits[n] for n in range(3, 7))),
        "second_zone_hits": second_hits,
        "second_zone_hit_rate": rate(second_hits),
        "method": "Walk-forward; every target uses only prior draws",
    }


def main():  # pragma: no cover - exercised by workflow CLI smoke tests
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()
    draws = load(args.csv)
    print(json.dumps(backtest(draws) if args.backtest else predict(draws),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
