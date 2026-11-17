#!/usr/bin/env python3
"""Reject reintroduction of retired independent 100-suite feature machinery.

Important: this is *not* a ban on OpenFE's normal candidate generation,
100/200/500 walk-forward validation folds, or unrelated numeric values.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PREFIXES = ("verification/of_lgbm_", ".github/workflows/of-lgbm-")
ACTIVE_FILES = {
    "WP-OS-Git-MAX/v6_openfe_bridge.py",
    "WP-OS-Git-MAX/v6_exact_e2e.py",
    ".github/workflows/v6-openfe-integration.yml",
}
# Match module-specific names, not generic OpenFE feature candidates.
RETIRED = re.compile(
    r"(?i)(?:100[_ -]?(?:independent[_ -]?)?(?:candidate[_ -]?)?"
    r"(?:feature[_ -]?)?(?:sets?|suites?|combinations?)"
    r"|(?:independent[_ -]?)?(?:candidate[_ -]?)?(?:feature[_ -]?)?"
    r"(?:sets?|suites?|combinations?)[_ -]?100|100套(?:獨立)?(?:候選)?特徵)"
)


def active_path(name: str) -> bool:
    if name == Path(__file__).relative_to(ROOT).as_posix():
        return False
    return name.startswith(ACTIVE_PREFIXES) or name in ACTIVE_FILES


def main() -> None:
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).decode("utf-8").split("\0")
    findings = []
    scanned = 0
    for name in filter(active_path, tracked):
        path = ROOT / name
        if not path.is_file():
            continue
        scanned += 1
        if RETIRED.search(name):
            findings.append(f"{name}: retired module filename")
        if path.suffix not in {".py", ".yml", ".yaml", ".json", ".toml", ".txt"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            if RETIRED.search(line):
                findings.append(f"{name}:{i}: retired feature suite reference")
    if findings:
        raise SystemExit("Retired independent 100 feature suites detected:\n" +
                         "\n".join(findings))
    print(f"PASS: {scanned} active files contain no retired independent 100-suite references")


if __name__ == "__main__":
    main()
