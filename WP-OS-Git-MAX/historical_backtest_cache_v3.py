#!/usr/bin/env python3
"""Versioned full-history walk-forward backtest cache for all WP-OS models."""
import argparse
import json
from pathlib import Path

from cache_and_hit_validation import digest, load_draws
from wp_os_git_max import backtest as backtest_main
from wp_os_git_max_10_20 import backtest as backtest_control
from wp_os_git_max_fusion import backtest as backtest_fusion

SCHEMA_VERSION = 3
MODEL_VERSIONS = {
    "main_5_10": "wp-os-git-max-5-10-v1",
    "control_10_20": "wp-os-git-max-10-20-control-v1",
    "fusion_5_10_20": "wp-os-git-max-fusion-5-10-20-v1",
}


def compute(draws):
    return {
        "main_5_10": backtest_main(draws),
        "control_10_20": backtest_control(draws),
        "fusion_5_10_20": backtest_fusion(draws),
    }


def run(csv_path, cache_path):
    draws = load_draws(csv_path)
    if len(draws) < 20:
        raise ValueError("At least twenty historical draws required")
    key = {
        "cache_schema_version": SCHEMA_VERSION,
        "model_versions": MODEL_VERSIONS,
        "first_draw": draws[0]["draw_id"],
        "through_draw": draws[-1]["draw_id"],
        "draw_count": len(draws),
        "archive_sha256": digest(draws),
    }
    path = Path(cache_path)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            unsigned = {k: v for k, v in stored.items() if k != "sha256"}
            intact = stored.get("sha256") == digest(unsigned)
            if intact and all(stored.get(k) == value for k, value in key.items()):
                results = stored.get("backtests")
                if isinstance(results, dict) and set(results) == set(MODEL_VERSIONS):
                    return {"cache": "hit", **key, "backtests": results}
        except (ValueError, OSError, TypeError, AttributeError):
            pass
    results = compute(draws)
    stored = {**key, "backtests": results}
    stored["sha256"] = digest(stored)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(stored, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {"cache": "refreshed", **key, "backtests": results}


def main():  # pragma: no cover - workflow exercises the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.cache), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
