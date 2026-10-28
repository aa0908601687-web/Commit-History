#!/usr/bin/env python3
"""ABE hybrid entry point: honest fallback until four neural adapters exist."""
import argparse
import hashlib
import json
from pathlib import Path

from adaptive_bayesian_ensemble_max import base, predict

CONFIG = Path(__file__).with_name("abe_ultra_hybrid_max.json")


def run(draws):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if len(draws) < config["data_window_draws"]:
        raise ValueError("At least 20 completed draws required")
    unavailable = [name for name, state in config["components"].items()
                   if name != "abe_bayesian" and state != "implemented_and_verified"]
    if config["components"]["abe_bayesian"] not in (
        "implemented_and_verified", "existing_standalone_implementation_not_yet_connected_to_fusion"
    ):
        raise RuntimeError("ABE core unavailable")
    # Do not silently renormalize 60% into a falsely labeled five-model forecast.
    baseline = predict(draws)
    result = {
        "system": config["name"],
        "execution_mode": "abe_only_fallback",
        "full_five_model_fusion_executed": False,
        "through_draw": baseline["through_draw"],
        "first": baseline["first"],
        "second": baseline["second"],
        "initial_weights": config["initial_weights"],
        "unavailable_models": unavailable,
        "abe_core_weights": baseline["weights"],
        "note": "Outputs are from the existing ABE core only, not t0-gamma/LSTM/GRU/Chronos-2."
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    print(json.dumps(run(base.load(args.csv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
