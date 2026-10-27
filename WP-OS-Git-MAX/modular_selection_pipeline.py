#!/usr/bin/env python3
"""Unified orchestration and configurable feature checks for Power Lottery models."""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from wp_os_git_max import load, predict as main_predict
from wp_os_git_max_10_20 import predict as control_predict
from wp_os_git_max_fusion import predict as fusion_predict

PIPELINE_VERSION = "wp-modular-pipeline-v2"
NAMES = ("main_5_10", "control_10_20", "fusion_5_10_20")
CHECK_NAMES = ("odd_even_ratio", "sum_range", "hot_cold_balance")
DEFAULT_CHECKS = {
    "odd_even_ratio": {"min_odd": 2, "max_odd": 4},
    "sum_range": {"minimum": 75, "maximum": 165},
    "hot_cold_balance": {"hot_pool_size": 12, "min_hot": 1, "max_hot": 4},
}


def _feature_checks(numbers, draws, enabled_checks=None):
    """Return transparent pass/fail diagnostics without changing model weights."""
    enabled = set(CHECK_NAMES if enabled_checks is None else enabled_checks)
    unknown = enabled - set(CHECK_NAMES)
    if unknown:
        raise ValueError("Unknown feature checks: " + ", ".join(sorted(unknown)))

    odd = sum(number % 2 for number in numbers)
    total = sum(numbers)
    counts = Counter(number for draw in draws[-10:] for number in draw["first"])
    hot_pool = {
        number for number, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:DEFAULT_CHECKS["hot_cold_balance"]["hot_pool_size"]]
    }
    hot_count = len(set(numbers) & hot_pool)
    values = {
        "odd_even_ratio": {
            "enabled": "odd_even_ratio" in enabled,
            "passed": DEFAULT_CHECKS["odd_even_ratio"]["min_odd"] <= odd
                      <= DEFAULT_CHECKS["odd_even_ratio"]["max_odd"],
            "observed": {"odd": odd, "even": 6 - odd},
            "rule": DEFAULT_CHECKS["odd_even_ratio"],
        },
        "sum_range": {
            "enabled": "sum_range" in enabled,
            "passed": DEFAULT_CHECKS["sum_range"]["minimum"] <= total
                      <= DEFAULT_CHECKS["sum_range"]["maximum"],
            "observed": total,
            "rule": DEFAULT_CHECKS["sum_range"],
        },
        "hot_cold_balance": {
            "enabled": "hot_cold_balance" in enabled,
            "passed": DEFAULT_CHECKS["hot_cold_balance"]["min_hot"] <= hot_count
                      <= DEFAULT_CHECKS["hot_cold_balance"]["max_hot"],
            "observed": {"hot": hot_count, "cold": 6 - hot_count},
            "rule": DEFAULT_CHECKS["hot_cold_balance"],
        },
    }
    applicable = [item["passed"] for item in values.values() if item["enabled"]]
    return {"accepted": all(applicable), "checks": values}


def _adjust_candidate(prediction, draws, enabled_checks=None):
    """Choose the highest-scoring compliant six-number set from the top ten."""
    original = list(prediction["first"])
    original_check = _feature_checks(original, draws, enabled_checks)
    if original_check["accepted"]:
        return {
            "first": original,
            "changed": False,
            "source": "original_prediction",
            "feature_validation": original_check,
        }

    scores = dict(prediction["top_first_scores"])
    pool = sorted(scores, key=lambda number: (-scores[number], number))[:10]
    candidates = []
    for combination in itertools.combinations(pool, 6):
        validation = _feature_checks(combination, draws, enabled_checks)
        if validation["accepted"]:
            candidates.append((sum(scores[number] for number in combination),
                               tuple(sorted(combination)), validation))
    if not candidates:
        return {
            "first": original,
            "changed": False,
            "source": "no_compliant_top10_candidate",
            "feature_validation": original_check,
        }
    _, selected, validation = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return {
        "first": list(selected),
        "changed": list(selected) != original,
        "source": "highest_score_compliant_top10",
        "feature_validation": validation,
    }


def run(draws, enabled_checks=None, enforce=False):
    if len(draws) < 20:
        raise ValueError("Modular pipeline requires at least 20 completed draws")
    # Each existing model receives the same cutoff; feature checks have 0% score weight.
    predictions = {
        "main_5_10": main_predict(draws),
        "control_10_20": control_predict(draws),
        "fusion_5_10_20": fusion_predict(draws),
    }
    cutoff = draws[-1]["draw_id"]
    feature_validation = {}
    adjusted_predictions = {}
    for name, prediction in predictions.items():
        if prediction["through_draw"] != cutoff:
            raise ValueError("Cutoff mismatch: " + name)
        first, second = prediction["first"], prediction["second"]
        if (len(first) != 6 or len(set(first)) != 6
                or not all(isinstance(x, int) and 1 <= x <= 38 for x in first)
                or not isinstance(second, int) or not 1 <= second <= 8):
            raise ValueError("Invalid prediction: " + name)
        feature_validation[name] = _feature_checks(first, draws, enabled_checks)
        adjusted = _adjust_candidate(prediction, draws, enabled_checks)
        adjusted_predictions[name] = {
            **adjusted,
            "second": second,
        }
        if enforce and not feature_validation[name]["accepted"]:
            raise ValueError("Feature checks rejected prediction: " + name)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "through_draw": cutoff,
        "source_draws": len(draws),
        "predictions": predictions,
        "adjusted_predictions": adjusted_predictions,
        "feature_checks": {
            "mode": "enforce" if enforce else "report_only",
            "score_weight": 0.0,
            "models": feature_validation,
        },
        "first_zone_overlap": {
            "main_control": sorted(set(predictions["main_5_10"]["first"]) & set(predictions["control_10_20"]["first"])),
            "main_fusion": sorted(set(predictions["main_5_10"]["first"]) & set(predictions["fusion_5_10_20"]["first"])),
            "control_fusion": sorted(set(predictions["control_10_20"]["first"]) & set(predictions["fusion_5_10_20"]["first"])),
        },
        "note": "Research outputs only; filters and model agreement do not improve lottery odds.",
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def main():  # pragma: no cover - exercised by Actions CLI parity check
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--disable-check", action="append", choices=CHECK_NAMES, default=[])
    parser.add_argument("--enforce-checks", action="store_true")
    args = parser.parse_args()
    enabled = [name for name in CHECK_NAMES if name not in args.disable_check]
    report = run(load(args.csv), enabled_checks=enabled, enforce=args.enforce_checks)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":  # pragma: no cover
    main()
