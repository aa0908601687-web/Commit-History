#!/usr/bin/env python3
"""Partial live-trained ABE+LSTM+GRU score bridge; unimplemented models remain fair priors."""
import argparse
import json
from pathlib import Path

from adaptive_bayesian_ensemble_max import base, predict, normalized, rank_zone
from abe_score_adapters import validate_scores


def bridge(draws, partial):
    cutoff = draws[-1]["draw_id"]
    if partial.get("through_draw") != cutoff or set(partial.get("models", {})) != {"lstm", "gru"}:
        raise ValueError("Expected exactly trained LSTM and GRU at matching cutoff")
    # Validate the two trained outputs via the shared adapter schema.
    # Dummy slots are used ONLY for checking shapes, never as network predictions.
    neutral = {zone: {str(i): 1 for i in range(1, upper+1)}
               for zone, upper in (("first", 38), ("second", 8))}
    dummy = {"first": neutral["first"], "second": neutral["second"]}
    validated = validate_scores(
        {"through_draw": cutoff,
         "models": {**partial["models"], "chronos_2": dummy, "t0_gamma": dummy}}, cutoff)
    baseline = predict(draws)
    result = {}
    for zone, upper, scalar in (("first", 38, False), ("second", 8, True)):
        seq = [d[zone] for d in draws]
        w = baseline["weights"][zone]["5_10"]
        a = normalized(rank_zone(seq, upper, 5, 10, scalar))
        b = normalized(rank_zone(seq, upper, 10, 20, scalar))
        combined = {
            n: .6 * (w*a[n] + (1-w)*b[n])
               + .1 * validated["lstm"][zone][n]
               + .1 * validated["gru"][zone][n]
               + .2 / upper
            for n in range(1,upper+1)
        }
        chosen = sorted(combined, key=lambda n:(-combined[n],n))[:1 if scalar else 6]
        result[zone] = chosen[0] if scalar else sorted(chosen)
    return {
        "through_draw": cutoff, **result,
        "execution_mode": "abe_lstm_gru_partial_with_uniform_reserve",
        "components_executed": ["abe_bayesian", "lstm", "gru"],
        "not_executed": ["chronos_2", "t0_gamma"],
        "effective_weights": {"abe_bayesian": .6, "lstm": .1, "gru": .1,
                              "unallocated_uniform_reserve": .2},
        "full_five_model_fusion_executed": False,
        "note": "20% missing-model reserve is uniform, NOT simulated neural output."
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--trained-scores", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    payload = json.loads(Path(args.trained_scores).read_text(encoding="utf-8"))
    result = bridge(base.load(args.csv), payload)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Partial real-trained neural score bridge completed; not full five models.")


if __name__ == "__main__":
    main()
