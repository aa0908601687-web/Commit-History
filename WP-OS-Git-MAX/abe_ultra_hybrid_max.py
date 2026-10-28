#!/usr/bin/env python3
"""ABE hybrid entry point: honest fallback until four neural adapters exist."""
import argparse
import hashlib
import json
from pathlib import Path

from adaptive_bayesian_ensemble_max import base, predict, rank_zone, normalized
from abe_score_adapters import read_scores, fuse

CONFIG = Path(__file__).with_name("abe_ultra_hybrid_max.json")


def run(draws, model_scores_path=None):
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
    # Hash the complete ordered input used by adaptive weight fitting and the
    # latest feature window separately; neither hash proves pre-draw timing.
    ordered_draws = [{"draw_id": str(d["draw_id"]), "first": list(d["first"]),
                      "second": d["second"]} for d in draws]
    full_input_sha256 = hashlib.sha256(json.dumps(
        ordered_draws, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    feature_window_sha256 = hashlib.sha256(json.dumps(
        ordered_draws[-config["data_window_draws"]:], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    result = {
        "system": config["name"],
        "execution_mode": "abe_only_fallback",
        "full_five_model_fusion_executed": False,
        "through_draw": baseline["through_draw"],
        "source_draw_count": len(draws),
        "feature_window_draws": config["data_window_draws"],
        "full_input_sha256": full_input_sha256,
        "feature_window_sha256": feature_window_sha256,
        "input_hash_timestamp_proof": False,
        "first": baseline["first"],
        "second": baseline["second"],
        "initial_weights": config["initial_weights"],
        "unavailable_models": unavailable,
        "abe_core_weights": baseline["weights"],
        "note": "Outputs are from the existing ABE core only, not t0-gamma/LSTM/GRU/Chronos-2."
    }
    if model_scores_path is not None:
        external = read_scores(model_scores_path, baseline["through_draw"])
        abe_maps = {}
        for zone, upper, scalar in (("first", 38, False), ("second", 8, True)):
            seq = [d[zone] for d in draws]
            w = baseline["weights"][zone]["5_10"]
            a = normalized(rank_zone(seq, upper, 5, 10, scalar))
            b = normalized(rank_zone(seq, upper, 10, 20, scalar))
            abe_maps[zone] = {n: w * a[n] + (1-w) * b[n] for n in a}
        picks = fuse(abe_maps, external, config["initial_weights"])
        result["first"], result["second"] = picks["first"], picks["second"]
        result["execution_mode"] = "five_model_external_scores"
        result["full_five_model_fusion_executed"] = True
        result["unavailable_models"] = []
        result["note"] = ("Four external score tables provided and validated; "
                          "neural runtime execution and provenance are NOT verified.")
        result["model_scores_sha256"] = hashlib.sha256(
            Path(model_scores_path).read_bytes()).hexdigest()
    result["sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-scores", help="Verified, externally generated four-model score JSON")
    args = parser.parse_args()
    print(json.dumps(run(base.load(args.csv), model_scores_path=args.model_scores), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
