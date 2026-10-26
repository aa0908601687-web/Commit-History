#!/usr/bin/env python3
"""Independent 10/20-draw control model for WP-OS Git MAX research."""
import argparse
import hashlib
import json
from collections import Counter

from wp_os_git_max import load

MODEL_VERSION = "wp-os-git-max-10-20-control-v1"


def rank_draws(draws, upper):
    if len(draws) < 20:
        raise ValueError("Need 20 completed draws")
    short, long = draws[-10:], draws[-20:]
    count10 = Counter(n for draw in short for n in draw)
    count20 = Counter(n for draw in long for n in draw)
    last = set(long[-1])
    ranked = []
    for number in range(1, upper + 1):
        age = next((i for i, draw in enumerate(reversed(long)) if number in draw), 20)
        score = (.45 * count10[number] / 10 + .25 * count20[number] / 20
                 + .15 * (count20[number] + 1) / (20 + upper)
                 + .10 * (number in last) + .05 * min(age, 20) / 20)
        ranked.append((number, round(score, 12)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def rank_second(values, upper=8):
    if len(values) < 20:
        raise ValueError("Need 20 completed draws")
    short, long = values[-10:], values[-20:]
    count10, count20 = Counter(short), Counter(long)
    ranked = []
    for number in range(1, upper + 1):
        age = next((i for i, value in enumerate(reversed(long)) if value == number), 20)
        score = (.45 * count10[number] / 10 + .25 * count20[number] / 20
                 + .15 * (count20[number] + 1) / (20 + upper)
                 + .10 * (number == long[-1]) + .05 * min(age, 20) / 20)
        ranked.append((number, round(score, 12)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def predict(draws):
    if len(draws) < 20:
        raise ValueError("At least 20 completed draws are required")
    first_scores = rank_draws([draw["first"] for draw in draws], 38)
    second_scores = rank_second([draw["second"] for draw in draws])
    payload = {
        "model_version": MODEL_VERSION,
        "through_draw": draws[-1]["draw_id"],
        "selection_window": [10, 20],
        "first": sorted(number for number, _ in first_scores[:6]),
        "second": second_scores[0][0],
        "top_first_scores": first_scores[:10],
        "second_scores": second_scores,
        "method": "Independent 10/20-draw control; no predictive claim",
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
        "feature_windows": [10, 20],
        "first_zone_hits": {str(n): hits[n] for n in range(7)},
        "first_zone_3plus_rate": rate(sum(hits[n] for n in range(3, 7))),
        "second_zone_hits": second_hits,
        "second_zone_hit_rate": rate(second_hits),
        "method": "Walk-forward; each target uses only prior 10/20 draws",
    }


def compare(main_prediction, control_prediction):
    main_first = set(main_prediction["first"])
    control_first = set(control_prediction["first"])
    return {
        "through_draw": control_prediction["through_draw"],
        "main_window": [5, 10],
        "control_window": [10, 20],
        "main_first": main_prediction["first"],
        "main_second": main_prediction["second"],
        "control_first": control_prediction["first"],
        "control_second": control_prediction["second"],
        "first_overlap": sorted(main_first & control_first),
        "first_overlap_count": len(main_first & control_first),
        "second_same": main_prediction["second"] == control_prediction["second"],
    }


def main():  # pragma: no cover - exercised by workflow CLI smoke tests
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()
    draws = load(args.csv)
    result = backtest(draws) if args.backtest else predict(draws)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
