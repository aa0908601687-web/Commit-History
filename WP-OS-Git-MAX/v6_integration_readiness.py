#!/usr/bin/env python3
"""Fail-closed, non-mutating V6 integration readiness audit.

A successful three-model audit is NOT a claim of live API or exact integration.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))


def check():
    from wp_os_git_max import load, predict as main_predict
    from wp_os_git_max_10_20 import predict as control_predict
    from wp_os_git_max_fusion import predict as fusion_predict
    from modular_selection_pipeline import run

    data = ROOT / "historical_archive.csv"
    draws = load(str(data))
    if len(draws) < 20:
        raise ValueError("At least 20 stored prior draws required")
    cutoff = draws[-1]["draw_id"]
    predictions = {
        "main_5_10": main_predict(draws),
        "control_10_20": control_predict(draws),
        "fusion_5_10_20": fusion_predict(draws),
    }
    integrated = run(draws)
    for name, result in predictions.items():
        if result != integrated["predictions"][name] or result["through_draw"] != cutoff:
            raise AssertionError("Three-model parity/cutoff mismatch: " + name)
    return {
        "data": {"archive_count": len(draws), "cutoff": cutoff,
                 "sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        "three_model_parity": "pass",
        "official_api": "offline contract test only; live fetch not checked",
        "exact": "not verified in main; PR #23 requires separate review",
        "trained_ai": "not verified",
        "full_e2e": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-v6", action="store_true",
                        help="Return failure unless full V6 integration has independent evidence")
    args = parser.parse_args()
    report = check()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    if args.strict_v6 and not report["full_e2e"]:
        raise SystemExit("V6 strict gate NOT READY: live API + exact + trained AI unverified")


if __name__ == "__main__":
    main()
