#!/usr/bin/env python3
"""Three-layer cache runner: validated draw data, features, and deterministic prediction."""
import argparse
import hashlib
import json
from pathlib import Path
from cache_and_hit_validation import load_draws, get_versioned_cache, digest
from wp_os_git_max import predict

SCHEMA_VERSION = 3
MODEL_VERSION = "wp-os-git-max-5-10-v1"

def result_cache(draws, features, path):
    """Reuse only an intact result matching the current last-ten data and feature digest."""
    last10 = draws[-10:]
    key = {
        "cache_schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "through_draw": last10[-1]["draw_id"],
        "history_sha256": digest(last10),
        "feature_sha256": features["sha256"],
    }
    path = Path(path)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            intact = stored.get("sha256") == digest({k: v for k, v in stored.items() if k != "sha256"})
            if intact and all(stored.get(k) == v for k, v in key.items()):
                result = stored.get("prediction")
                if isinstance(result, dict) and result.get("sha256") == hashlib.sha256(
                    json.dumps({k:v for k,v in result.items() if k != "sha256"}, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest() and result.get("through_draw") == key["through_draw"]:
                    # JSON converts score tuples to lists; restore the public predict() shape.
                    result["top_first_scores"] = [tuple(x) for x in result["top_first_scores"]]
                    result["second_scores"] = [tuple(x) for x in result["second_scores"]]
                    return result, "hit"
        except (ValueError, OSError, TypeError, AttributeError):
            pass
    result = predict(draws)
    # Refuse to write a result that does not match the source history.
    if result["through_draw"] != key["through_draw"]:
        raise ValueError("Prediction source mismatch")
    stored = {**key, "prediction": result}
    stored["sha256"] = digest(stored)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stored, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return result, "refreshed"

def run(csv_path, feature_path, prediction_path, verify=False):
    draws = load_draws(csv_path)
    if len(draws) < 10:
        raise ValueError("At least ten prior draws required")
    features, feature_status = get_versioned_cache(draws, feature_path)
    result, prediction_status = result_cache(draws, features, prediction_path)
    if verify and result != predict(draws):
        raise ValueError("Cached and direct predictions differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "draws_stored": len(draws),
        "selection_window": [5, 10],
        "feature_cache": feature_status,
        "prediction_cache": prediction_status,
        "prediction": result,
        "consistency_verified": bool(verify),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--prediction-cache", required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.csv, args.feature_cache, args.prediction_cache, args.verify), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
