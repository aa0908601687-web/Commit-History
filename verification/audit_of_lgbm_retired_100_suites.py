#!/usr/bin/env python3
"""Repository-wide guard against retired independent 100-feature-suite machinery.

This guard deliberately permits OpenFE's native generated candidate features,
100/200/500 historical evaluation folds, and unrelated numeric constants.
It checks every tracked code/config/workflow file, not just OF-LGBM filenames.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).relative_to(ROOT).as_posix()
SCANNED_EXTENSIONS = {
    ".py", ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg",
    ".txt", ".sh", ".bash", ".ps1", ".xml",
}
# Named legacy suite/config/API identifiers, not OpenFE internals.
RETIRED = re.compile(
    r"(?i)(?:"
    r"100[_ -]?(?:independent[_ -]?)?(?:candidate[_ -]?)?"
    r"(?:feature[_ -]?)?(?:sets?|suites?|combinations?)"
    r"|(?:independent[_ -]?)?(?:candidate[_ -]?)?(?:feature[_ -]?)?"
    r"(?:sets?|suites?|combinations?)[_ -]?100"
    r"|100套(?:獨立)?(?:候選)?特徵"
    r")"
)


def inspect_files(paths: list[str]) -> tuple[int, list[str]]:
    findings: list[str] = []
    scanned = 0
    for name in paths:
        if name == SELF:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        if RETIRED.search(name):
            findings.append(f"{name}: retired filename")
        if path.suffix.lower() not in SCANNED_EXTENSIONS:
            continue
        scanned += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeError, OSError) as exc:
            findings.append(f"{name}: cannot inspect: {exc}")
            continue
        for number, line in enumerate(lines, 1):
            # A workflow may legitimately reference this guard script by filename.
            # Ignore only that exact self-reference; inspect all other text.
            inspected_line = line.replace(SELF, "")
            if RETIRED.search(inspected_line):
                findings.append(f"{name}:{number}: retired reference")
    return scanned, findings


def main() -> None:
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).decode("utf-8").split("\0")
    scanned, findings = inspect_files([p for p in tracked if p])
    if findings:
        raise SystemExit(
            "FAIL: retired independent 100 feature suites detected:\n"
            + "\n".join(findings)
        )
    print(
        f"PASS: scanned {scanned} tracked code/config/CI files; "
        "no retired independent 100-suite references"
    )


if __name__ == "__main__":
    main()
