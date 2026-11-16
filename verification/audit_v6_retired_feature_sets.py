#!/usr/bin/env python3
"""Audit active V6 source for retired independent 100 candidate feature suites.

This audit does not prohibit OpenFE-generated features or references to 100
historical draws, percentages, or unrelated legacy files.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = (
    "WP-OS-Git-MAX/v6_openfe_bridge.py",
    "WP-OS-Git-MAX/v6_exact_e2e.py",
    "WP-OS-Git-MAX/v6_exact_search.py",
    ".github/workflows/v6-postmerge-readonly-e2e.yml",
)
# Explicit names for a fixed 100-suite selector; don't match ordinary 100%.
RETIRED = re.compile(
    r"(?i)(?:100[_ -]?(?:candidate[_ -]?)?feature[_ -]?(?:sets?|suites?|combinations?)"
    r"|(?:independent[_ -]?)?feature[_ -]?(?:sets?|suites?|combinations?)[_ -]?100"
    r"|(?:candidate[_ -]?)?feature[_ -]?combinations?\s*[:=]\s*100)"
)

def audit():
    failures = []
    for rel in ACTIVE:
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing active file: {rel}")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RETIRED.search(line):
                failures.append(f"{rel}:{lineno}: retired 100-suite reference")
    return failures

if __name__ == "__main__":
    failures = audit()
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)
    print("PASS: no explicit retired independent 100-feature-suite references in active V6 entrypoints")
    print("Scope: " + ", ".join(ACTIVE))
    print("This is a scoped static audit, NOT proof of whole-repository removal.")
