#!/usr/bin/env python3
"""Validate the pinned superintelligence upstream checkout without executing it.

This is a provenance/readiness boundary only. It does not feed lottery features,
weights, predictions, or caches.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

UPSTREAM_REPOSITORY = "marciolscoutinho/superintelligence"
PINNED_COMMIT = "da09a58ae32aed50cdf13caa8a5e2f4c795a324d"
REQUIRED_FILES = (
    "LICENSE",
    "main.py",
    "agents/base_agent.py",
    "models/auto_ml.py",
    "auto_improvement/auto_ml_manager.py",
    "auto_improvement/adaptive_learning.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate(root: Path, expected_commit: str = PINNED_COMMIT) -> dict:
    root = root.resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    syntax_errors: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file():
            continue
        hashes[name] = _sha256(path)
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                syntax_errors[name] = str(exc)

    head = _git_head(root)
    commit_matches = head == expected_commit if head is not None else None
    structure_ok = not missing and not syntax_errors
    pinned_checkout_ok = structure_ok and commit_matches is True
    return {
        "schema_version": 1,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "expected_commit": expected_commit,
        "observed_commit": head,
        "commit_matches": commit_matches,
        "required_files": list(REQUIRED_FILES),
        "missing_files": missing,
        "syntax_errors": syntax_errors,
        "file_sha256": hashes,
        "structure_ok": structure_ok,
        "pinned_checkout_ok": pinned_checkout_ok,
        "connection_status": "provenance_connected" if pinned_checkout_ok else "failed",
        "execution_status": "not_executed",
        "prediction_integration": False,
        "predictive_advantage_verified": False,
        "note": (
            "Static provenance/readiness validation only; upstream code is not "
            "executed and does not enter WP-OS predictions or caches."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", default=PINNED_COMMIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.upstream_dir, args.expected_commit)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["pinned_checkout_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
