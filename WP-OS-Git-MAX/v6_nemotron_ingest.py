#!/usr/bin/env python3
"""Opt-in Nemotron extraction audit before V6 OF3Q inference.

This adapter accepts *already extracted* structured JSON (e.g. prepared by a
vision-language model). It does NOT install, call, or claim to run Nemotron.
It never changes selection weights or the canonical CSV.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from v6_exact_e2e import run, validate


def audit_extraction(csv_path, extraction_path):
    """Fail closed: all extracted draw values must match trusted canonical CSV."""
    canonical_path = Path(csv_path)
    extraction_path = Path(extraction_path)
    canonical = {draw_id: (numbers, second) for draw_id, numbers, second in validate(canonical_path)}
    data = json.loads(extraction_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("draws"), list) or not data["draws"]:
        raise ValueError("extraction must contain a nonempty draws list")
    if not isinstance(data.get("source"), str) or not data["source"].strip():
        raise ValueError("extraction source is required")
    seen = set()
    for item in data["draws"]:
        if not isinstance(item, dict) or set(item) != {"draw_id", "first", "second"}:
            raise ValueError("each draw must contain exactly draw_id, first, second")
        draw_id, first, second = item["draw_id"], item["first"], item["second"]
        if not isinstance(draw_id, str) or len(draw_id) != 9 or not draw_id.isdigit() or draw_id in seen:
            raise ValueError("invalid or duplicate extracted draw_id")
        seen.add(draw_id)
        if (not isinstance(first, list) or len(first) != 6 or
                any(type(n) is not int or not 1 <= n <= 38 for n in first) or
                len(set(first)) != 6 or type(second) is not int or not 1 <= second <= 8):
            raise ValueError("invalid extracted lottery numbers")
        if draw_id not in canonical:
            raise ValueError("extracted draw absent from trusted canonical CSV: " + draw_id)
        if (set(first), second) != (set(canonical[draw_id][0]), canonical[draw_id][1]):
            raise ValueError("Nemotron extraction mismatch: " + draw_id)
    return {
        "status": "matched_canonical_csv",
        "source": data["source"],
        "checked_draws": len(seen),
        "canonical_csv_sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
        "extraction_sha256": hashlib.sha256(extraction_path.read_bytes()).hexdigest(),
        "nemotron_model_executed": False,
        "note": "Audit of supplied model extraction only; no live OCR or Nemotron inference",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="trusted V6 CSV, unchanged")
    parser.add_argument("--nemotron-extraction", required=True, help="structured JSON output from external vision process")
    parser.add_argument("--openfe-three-views", action="store_true")
    parser.add_argument("--pytorch-three-views", action="store_true")
    parser.add_argument("--pytorch-weight", type=float, default=0.5)
    parser.add_argument("--cudaq", action="store_true")
    parser.add_argument("--require-cudaq", action="store_true")
    parser.add_argument("--ising-qubo-check", action="store_true")
    args = parser.parse_args()
    audit = audit_extraction(args.csv, args.nemotron_extraction)
    result = run(args.csv, openfe_three_views=args.openfe_three_views,
                 pytorch_three_views=args.pytorch_three_views,
                 pytorch_weight=args.pytorch_weight,
                 use_cudaq=args.cudaq or args.require_cudaq,
                 require_cudaq=args.require_cudaq,
                 ising_qubo_check=args.ising_qubo_check)
    result["nemotron_audit"] = audit
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
