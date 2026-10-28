#!/usr/bin/env python3
"""Optional evidence-gated selector. Original WP-OS models/weights remain unchanged.

Historical evaluation is exploratory, not a prospective blind test.
Only earlier completed draws are used when evaluating each target.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from wp_os_git_max import load, predict as predict_main
from wp_os_git_max_10_20 import predict as predict_control
from wp_os_git_max_fusion import predict as predict_fusion

MODELS = {"main_5_10": predict_main, "control_10_20": predict_control,
          "fusion_5_10_20": predict_fusion}
FAIR_3PLUS = sum(math.comb(6, k) * math.comb(32, 6-k)
                 for k in range(3, 7)) / math.comb(38, 6)


def history_report(draws, holdout=200):
    """Strictly chronological replay, with reserved last holdout for reporting.

    This is NOT an untouched holdout if developers have inspected it.
    """
    if len(draws) < 20 + holdout or holdout < 1:
        raise ValueError("Need 20 warmup draws and requested holdout")
    start = len(draws) - holdout
    report = {name: {"hits_0_to_6": [0]*7, "second_hits": 0}
              for name in MODELS}
    for end in range(start, len(draws)):
        seen = draws[:end]
        actual = draws[end]
        for name, model in MODELS.items():
            p = model(seen)
            hits = len(set(p["first"]) & set(actual["first"]))
            report[name]["hits_0_to_6"][hits] += 1
            report[name]["second_hits"] += int(p["second"] == actual["second"])
    for item in report.values():
        item["three_plus"] = sum(item["hits_0_to_6"][3:])
        item["three_plus_rate"] = item["three_plus"] / holdout
        item["second_rate"] = item["second_hits"] / holdout
    return {"periods": holdout, "through_draw": draws[-1]["draw_id"],
            "baseline_three_plus": FAIR_3PLUS,
            "warning": "Retrospective replay only; not independent prospective evidence",
            "models": report}


def evidence_gate(report):
    """Conservative default: without prospective independent proof abstain.

    Do not select a winner based merely on the best retrospective rate.
    """
    return {"selected_model": None, "status": "no_prospective_evidence",
            "reason": "Historical backtests alone cannot establish a repeatable edge"}


def fallback(draws):
    """Deterministic reproducible fair-format sample; no win-rate improvement."""
    cutoff = draws[-1]["draw_id"]
    # Hash is for reproducibility, NOT a timestamp or randomness guarantee.
    seed = int.from_bytes(hashlib.sha256(
        ("wp-evidence-gate-v1|" + cutoff).encode()).digest()[:16], "big")
    rng = random.Random(seed)
    return {"through_draw": cutoff,
            "first": sorted(rng.sample(range(1, 39), 6)),
            "second": rng.randint(1, 8),
            "method": "seeded_uniform_fallback",
            "warning": "Research selection, not a probability advantage"}


def run(draws, holdout=200):
    report = history_report(draws, holdout)
    gate = evidence_gate(report)
    return {"gate": gate, "candidate": fallback(draws), "history": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--holdout", type=int, default=200)
    args = ap.parse_args()
    print(json.dumps(run(load(args.csv), args.holdout), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
