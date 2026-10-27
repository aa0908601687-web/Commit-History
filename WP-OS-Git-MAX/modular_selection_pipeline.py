#!/usr/bin/env python3
"""Non-mutating unified orchestration of three independent Power Lottery models."""
import argparse
import hashlib
import json
from pathlib import Path

from wp_os_git_max import load, predict as main_predict
from wp_os_git_max_10_20 import predict as control_predict
from wp_os_git_max_fusion import predict as fusion_predict

PIPELINE_VERSION = "wp-modular-pipeline-v1"
NAMES = ("main_5_10", "control_10_20", "fusion_5_10_20")


def run(draws):
    if len(draws) < 20:
        raise ValueError("Modular pipeline requires at least 20 completed draws")
    # Each existing model receives the same historical cutoff, without altering weights.
    predictions = {
        "main_5_10": main_predict(draws),
        "control_10_20": control_predict(draws),
        "fusion_5_10_20": fusion_predict(draws),
    }
    cutoff = draws[-1]["draw_id"]
    for name, prediction in predictions.items():
        if prediction["through_draw"] != cutoff:
            raise ValueError("Cutoff mismatch: " + name)
        first, second = prediction["first"], prediction["second"]
        if (len(first) != 6 or len(set(first)) != 6
                or not all(isinstance(x, int) and 1 <= x <= 38 for x in first)
                or not isinstance(second, int) or not 1 <= second <= 8):
            raise ValueError("Invalid prediction: " + name)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "through_draw": cutoff,
        "source_draws": len(draws),
        "predictions": predictions,
        "first_zone_overlap": {
            "main_control": sorted(set(predictions["main_5_10"]["first"]) & set(predictions["control_10_20"]["first"])),
            "main_fusion": sorted(set(predictions["main_5_10"]["first"]) & set(predictions["fusion_5_10_20"]["first"])),
            "control_fusion": sorted(set(predictions["control_10_20"]["first"]) & set(predictions["fusion_5_10_20"]["first"])),
        },
        "note": "Research outputs only; cache integrity and agreement do not improve lottery odds.",
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def main():  # pragma: no cover - exercised by Actions CLI parity check
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()
    report = run(load(args.csv))
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":  # pragma: no cover
    main()
