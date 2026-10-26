#!/usr/bin/env python3
"""WP-OS V3: source -> verified feature cache -> deterministic selection/snapshot."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from cache_and_hit_validation import digest, load_draws, validate
from wp_os_git_max import predict as legacy_predict

SCHEMA = 3
MODEL = "wp-os-git-max-5-10-v1"
WEIGHTS = (0.45, 0.25, 0.15, 0.10, 0.05)

def expected(draws):
    if len(draws) < 10:
        raise ValueError("Need ten completed draws")
    h = draws[-10:]
    def zone(key, upper):
        seq = [d[key] for d in h]
        def count(sample, n):
            return sum(n in d if key == "first" else n == d for d in sample)
        return {
            "five": [count(seq[-5:], n) for n in range(1, upper + 1)],
            "ten": [count(seq, n) for n in range(1, upper + 1)],
            "age": [next((i for i, d in enumerate(reversed(seq))
                         if (n in d if key == "first" else n == d)), 10)
                    for n in range(1, upper + 1)],
            "last": list(seq[-1]) if key == "first" else seq[-1],
        }
    pairs = Counter(p for d in h for p in itertools.combinations(d["first"], 2))
    obj = {"schema": SCHEMA, "model": MODEL, "through_draw": h[-1]["draw_id"],
           "source_sha256": digest(h), "first": zone("first", 38),
           "second": zone("second", 8),
           "pairs": [[list(p), c] for p, c in sorted(pairs.items(),
                                                     key=lambda x: (-x[1], x[0]))[:10]]}
    obj["sha256"] = digest(obj)
    return obj

def cache_for(draws, path):
    want = expected(draws)
    path = Path(path)
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("sha256") == digest({k:v for k,v in old.items() if k != "sha256"}) and old == want:
                return old, "hit"
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(want, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return want, "refreshed"

def rank(zone, upper):
    results = []
    for n in range(1, upper + 1):
        idx = n - 1
        a, b = zone["five"][idx], zone["ten"][idx]
        last = (n in zone["last"]) if upper == 38 else (n == zone["last"])
        s = (.45 * a / 5 + .25 * b / 10 + .15 * (b + 1) / (10 + upper)
             + .10 * last + .05 * min(zone["age"][idx], 10) / 10)
        results.append((n, round(s, 12)))
    return sorted(results, key=lambda x: (-x[1], x[0]))

def predict_from_cache(cache):
    if cache.get("schema") != SCHEMA or cache.get("model") != MODEL:
        raise ValueError("V3 cache version mismatch")
    if cache.get("sha256") != digest({k:v for k,v in cache.items() if k != "sha256"}):
        raise ValueError("V3 cache digest mismatch")
    a, b = rank(cache["first"], 38), rank(cache["second"], 8)
    prediction = {"through_draw": cache["through_draw"],
                  "first": sorted(n for n, _ in a[:6]),
                  "second": b[0][0], "top_first_scores": a[:10],
                  "second_scores": b,
                  "top_pairs": [{"pair":p, "count":c} for p,c in cache["pairs"]],
                  "method": "Fixed 5/10-draw research baseline, no predictive claim"}
    prediction["sha256"] = hashlib.sha256(json.dumps(
        prediction, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return prediction

def snapshot(draws, prediction, target):
    if not (target.isdigit() and len(target) == 9 and target > draws[-1]["draw_id"]):
        raise ValueError("Target must be a later 9-digit draw ID")
    obj = {"target_draw": target, "through_draw": draws[-1]["draw_id"],
           "history_sha256": digest(draws[-10:]), "first": prediction["first"],
           "second": prediction["second"], "model_prediction_sha256": prediction["sha256"],
           "note": "Publish/commit before drawing; local hash alone cannot prove timing"}
    obj["sha256"] = digest(obj)
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prediction")
    parser.add_argument("--snapshot")
    parser.add_argument("--target-draw")
    parser.add_argument("--validate-snapshot")
    args = parser.parse_args()
    draws = load_draws(args.csv)
    cache, status = cache_for(draws, args.cache)
    result = predict_from_cache(cache)
    # Fail closed on any scoring drift against the original 5/10 model.
    if result != legacy_predict(draws):
        raise ValueError("V3 cached selection differs from original model")
    if args.prediction:
        Path(args.prediction).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.snapshot:
        if not args.target_draw:
            parser.error("--target-draw required with --snapshot")
        p = Path(args.snapshot)
        if p.exists():
            raise FileExistsError("Refusing to overwrite snapshot")
        p.write_text(json.dumps(snapshot(draws, result, args.target_draw),
                                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verified = None
    if args.validate_snapshot:
        verified = validate(json.loads(Path(args.validate_snapshot).read_text(encoding="utf-8")), draws)
    print(json.dumps({"cache_status": status, "schema": SCHEMA, "source_sha256": cache["source_sha256"],
                      "prediction": result, "validation": verified}, ensure_ascii=False))

if __name__ == "__main__":
    main()
