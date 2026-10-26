#!/usr/bin/env python3
"""Optional V3 three-layer reproducible cache for WP-OS Git MAX.

Retains the original 5/10 selection logic and never changes model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

from wp_os_git_max import load, predict
from cache_and_hit_validation import digest, get_versioned_cache

V3_SCHEMA = 3
V3_MODEL = "wp-os-git-max-5-10-v1"

def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def _verified(path, expected):
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        signature = stored.get("sha256")
        return (isinstance(signature, str)
                and signature == digest({k: v for k, v in stored.items() if k != "sha256"})
                and stored == expected)
    except (OSError, ValueError, TypeError, AttributeError):
        return False

def _signed(obj):
    result = dict(obj)
    result["sha256"] = digest(result)
    return result

def run(csv_path, cache_dir):
    draws = load(csv_path)
    if len(draws) < 10:
        raise ValueError("Need at least 10 draws")
    folder = Path(cache_dir)
    # Layer 1: complete loaded source fingerprint, including all 20 stored rows.
    source = _signed({
        "cache_schema_version": V3_SCHEMA,
        "layer": "draws",
        "draw_count": len(draws),
        "through_draw": draws[-1]["draw_id"],
        "history_sha256": digest(draws)
    })
    source_path = folder / "draws.json"
    source_hit = _verified(source_path, source)
    if not source_hit:
        _save(source_path, source)

    # Layer 2: validated original Schema V2 5/10 feature cache.
    feature, feature_status = get_versioned_cache(draws, folder / "features.json")

    # Layer 3: preserve exact original prediction output, recompute only on miss.
    prediction_path = folder / "prediction.json"
    if prediction_path.exists():
        try:
            old = json.loads(prediction_path.read_text(encoding="utf-8"))
            payload = old.get("prediction")
            unsigned = {k: v for k, v in old.items() if k != "sha256"}
            if (old.get("sha256") == digest(unsigned)
                and old.get("cache_schema_version") == V3_SCHEMA
                and old.get("model_version") == V3_MODEL
                and old.get("feature_sha256") == feature["sha256"]
                and old.get("through_draw") == draws[-1]["draw_id"]
                and isinstance(payload, dict)
                and payload.get("sha256") == hashlib.sha256(
                    json.dumps({k: v for k, v in payload.items() if k != "sha256"},
                               sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()):
                # JSON restores tuples as lists; recover the original public predict() shape.
                payload["top_first_scores"] = [tuple(item) for item in payload["top_first_scores"]]
                payload["second_scores"] = [tuple(item) for item in payload["second_scores"]]
                return {"prediction": payload, "cache": {"draws": "hit" if source_hit else "refreshed",
                         "features": feature_status, "prediction": "hit"}}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    payload = predict(draws)
    record = _signed({"cache_schema_version": V3_SCHEMA,
        "model_version": V3_MODEL, "feature_sha256": feature["sha256"],
        "through_draw": draws[-1]["draw_id"], "prediction": payload})
    _save(prediction_path, record)
    return {"prediction": payload, "cache": {"draws": "hit" if source_hit else "refreshed",
             "features": feature_status, "prediction": "refreshed"}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.csv, args.cache_dir), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
