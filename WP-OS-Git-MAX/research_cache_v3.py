#!/usr/bin/env python3
"""V3 result caches for the independent control and equal-weight fusion models."""
import argparse
import hashlib
import json
from pathlib import Path

from cache_and_hit_validation import digest, load_draws
from wp_os_git_max_10_20 import predict as predict_control
from wp_os_git_max_fusion import predict as predict_fusion

SCHEMA_VERSION = 3
MODELS = {
    "control_10_20": ("wp-os-git-max-10-20-control-v1", predict_control),
    "fusion_5_10_20": ("wp-os-git-max-fusion-5-10-20-v1", predict_fusion),
}


def _valid_prediction(result, through_draw):
    if not isinstance(result, dict) or result.get("through_draw") != through_draw:
        return False
    unsigned = {key: value for key, value in result.items() if key != "sha256"}
    expected = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return result.get("sha256") == expected


def result_cache(draws, path, model_version, predictor):
    recent = draws[-20:]
    key = {
        "cache_schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "through_draw": recent[-1]["draw_id"],
        "history_sha256": digest(recent),
    }
    path = Path(path)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            unsigned = {k: v for k, v in stored.items() if k != "sha256"}
            intact = stored.get("sha256") == digest(unsigned)
            if intact and all(stored.get(k) == value for k, value in key.items()):
                result = stored.get("prediction")
                if _valid_prediction(result, key["through_draw"]):
                    result["top_first_scores"] = [tuple(x) for x in result["top_first_scores"]]
                    result["second_scores"] = [tuple(x) for x in result["second_scores"]]
                    return result, "hit"
        except (ValueError, OSError, TypeError, AttributeError):
            pass
    result = predictor(draws)
    if not _valid_prediction(result, key["through_draw"]):
        raise ValueError("Prediction source or digest mismatch")
    stored = {**key, "prediction": result}
    stored["sha256"] = digest(stored)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(stored, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return result, "refreshed"


def run(csv_path, control_path, fusion_path, verify=False):
    draws = load_draws(csv_path)
    if len(draws) < 20:
        raise ValueError("At least twenty prior draws required")
    outputs = {}
    for name, path in (("control_10_20", control_path), ("fusion_5_10_20", fusion_path)):
        version, predictor = MODELS[name]
        prediction, status = result_cache(draws, path, version, predictor)
        if verify and prediction != predictor(draws):
            raise ValueError("Cached and direct predictions differ: " + name)
        outputs[name] = {"cache": status, "prediction": prediction}
    return {
        "schema_version": SCHEMA_VERSION,
        "draws_stored": len(draws),
        "selection_windows": [5, 10, 20],
        "models": outputs,
        "consistency_verified": bool(verify),
    }


def main():  # pragma: no cover - workflow exercises the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--control-cache", required=True)
    parser.add_argument("--fusion-cache", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.control_cache, args.fusion_cache, args.verify),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
