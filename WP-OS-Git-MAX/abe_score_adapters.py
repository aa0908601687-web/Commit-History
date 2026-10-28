"""Validated external model score adapter for future ABE ensemble integration.

Input JSON: {"through_draw": "...", "models": {"lstm": {
"first": {"1": 0.1, ... "38": 0.1}, "second": {"1": 0.1, ... "8": 0.1}
}}}
All 38+8 finite nonnegative scores must be present per participating model.
This module does NOT install or execute neural networks.
"""
import json
import math
from pathlib import Path

NAMES = ("lstm", "gru", "chronos_2", "t0_gamma")


def validate_scores(payload, cutoff):
    if not isinstance(payload, dict) or payload.get("through_draw") != cutoff:
        raise ValueError("Model output cutoff mismatch")
    raw = payload.get("models")
    if not isinstance(raw, dict) or set(raw) != set(NAMES):
        raise ValueError("All four distinct model outputs required")
    models = {}
    for name in NAMES:
        record = raw[name]
        if not isinstance(record, dict) or set(record) != {"first", "second"}:
            raise ValueError("Invalid zones for " + name)
        zones = {}
        for zone, upper in (("first", 38), ("second", 8)):
            scores = record[zone]
            if not isinstance(scores, dict) or set(scores) != {str(n) for n in range(1, upper + 1)}:
                raise ValueError("Incomplete " + name + " " + zone + " scores")
            vals = {n: scores[str(n)] for n in range(1, upper + 1)}
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0 for v in vals.values()):
                raise ValueError("Non-finite or negative score")
            total = sum(vals.values())
            if total <= 0 or not math.isfinite(total):
                raise ValueError("Scores must have positive finite sum")
            zones[zone] = {n: val / total for n, val in vals.items()}
        models[name] = zones
    return models


def read_scores(path, cutoff):
    return validate_scores(json.loads(Path(path).read_text(encoding="utf-8")), cutoff)


def fuse(abe_scores, other_scores, weights):
    """Fuse normalized ABE zone score maps with four validated adapters."""
    if not isinstance(weights, dict) or set(weights) != {"abe_bayesian", *NAMES}:
        raise ValueError("Invalid weight keys")
    if any(isinstance(w, bool) or not isinstance(w, (int, float))
           or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    if abs(math.fsum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to one")
    result = {}
    for zone, upper in (("first", 38), ("second", 8)):
        maps = {"abe_bayesian": abe_scores[zone],
                **{name: other_scores[name][zone] for name in NAMES}}
        combined = {}
        for n in range(1, upper + 1):
            combined[n] = sum(weights[name] * maps[name][n] for name in weights)
        ranks = sorted(combined, key=lambda n: (-combined[n], n))
        result[zone] = sorted(ranks[:6]) if zone == "first" else ranks[0]
    return result
