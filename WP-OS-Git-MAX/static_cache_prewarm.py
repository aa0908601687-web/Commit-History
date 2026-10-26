#!/usr/bin/env python3
"""Prebuild every supported WP-OS response and serve it cache-only.

This layer guarantees cache coverage for the finite profiles declared below.  It
does not claim that cached lottery selections are more accurate.
"""
import argparse
import json
from pathlib import Path

from cache_and_hit_validation import digest, load_draws

SCHEMA_VERSION = 1
MODEL_SET_VERSION = "wp-os-static-prewarm-v1"


class CacheMissError(LookupError):
    """Raised when a request is not present in the prebuilt cache."""


def _builders():
    from wp_os_git_max import backtest as main_backtest, predict as main_predict
    from wp_os_git_max_10_20 import backtest as control_backtest, predict as control_predict
    from wp_os_git_max_fusion import backtest as fusion_backtest, predict as fusion_predict

    return {
        "prediction/main_5_10": main_predict,
        "prediction/control_10_20": control_predict,
        "prediction/fusion_5_10_20": fusion_predict,
        "backtest/main_5_10": main_backtest,
        "backtest/control_10_20": control_backtest,
        "backtest/fusion_5_10_20": fusion_backtest,
    }


def build(draws, builders=None):
    if len(draws) < 20:
        raise ValueError("At least twenty completed draws are required")
    builders = builders or _builders()
    responses = {name: builder(draws) for name, builder in sorted(builders.items())}
    obj = {
        "cache_schema_version": SCHEMA_VERSION,
        "model_set_version": MODEL_SET_VERSION,
        "first_draw": draws[0]["draw_id"],
        "through_draw": draws[-1]["draw_id"],
        "draw_count": len(draws),
        "source_sha256": digest(draws),
        "profiles": sorted(responses),
        "responses": responses,
    }
    obj["sha256"] = digest(obj)
    return obj


def write_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_verified(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = obj.get("sha256")
    if checksum != digest({k: v for k, v in obj.items() if k != "sha256"}):
        raise ValueError("Static cache SHA-256 mismatch")
    if obj.get("cache_schema_version") != SCHEMA_VERSION or obj.get("model_set_version") != MODEL_SET_VERSION:
        raise ValueError("Static cache version mismatch")
    if obj.get("profiles") != sorted(obj.get("responses", {})):
        raise ValueError("Static cache profile manifest mismatch")
    return obj


def get_cache_only(cache, profile):
    """Return a prebuilt response; never fall through to calculation or source data."""
    try:
        return cache["responses"][profile]
    except (KeyError, TypeError) as exc:
        raise CacheMissError("Prebuilt cache miss: " + profile) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.prewarm:
        if not args.csv:
            parser.error("--csv is required with --prewarm")
        cache = build(load_draws(args.csv))
        write_atomic(cache, args.cache)
        # Re-open what was written so startup fails closed on corruption.
        cache = load_verified(args.cache)
        print(json.dumps({"status": "prewarmed", "profiles": cache["profiles"],
                          "through_draw": cache["through_draw"], "sha256": cache["sha256"]}, ensure_ascii=False))
        return
    cache = load_verified(args.cache)
    if not args.profile:
        parser.error("--profile is required for cache-only reads")
    print(json.dumps(get_cache_only(cache, args.profile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
