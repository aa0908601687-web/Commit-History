#!/usr/bin/env python3
"""Revalidate pinned GitHub Actions dated Power artifact into NON-PRODUCTION staging.

A passing run only establishes CSV/SQLite/canonical consistency. It cannot
establish independent Taiwan Lottery official provenance.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sqlite3
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

EXPECTED_ROWS = 1964
EXPECTED_ARTIFACT_ID = 14826352289
DATED_SHA256 = "84b26e6fbdf740dae94ce5105eb4c463c3dde14ae1f5bd5469b4801f753483de"
SQLITE_SHA256 = "308072c836008a85d266f0d45feec752de4ce3d447c5a670d7385c8d1cc64b8e"
ARCHIVE_SHA256 = "de92a6019d8efa300f4cbb0e96b7badd9babfea9889b5e7657f0b10a01a80e5b"
REPO = "aa0908601687-web/Commit-History"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def parse(data, dated):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig")), strict=True))
    required = {"draw_id", "first", "second"} | ({"draw_date"} if dated else set())
    if not rows or not required.issubset(rows[0]):
        raise ValueError("missing expected CSV columns")
    result = []
    last_id = -1
    last_date = None
    for line, row in enumerate(rows, 2):
        identifier = row["draw_id"]
        if len(identifier) != 9 or not identifier.isascii() or not identifier.isdecimal():
            raise ValueError(f"bad draw ID on line {line}")
        if int(identifier) <= last_id:
            raise ValueError(f"draw IDs not strictly increasing at line {line}")
        last_id = int(identifier)
        nums = tuple(sorted(int(n) for n in row["first"].replace(",", " ").split()))
        second = int(row["second"])
        if len(nums) != 6 or len(set(nums)) != 6 or not all(1 <= n <= 38 for n in nums):
            raise ValueError(f"bad Power first area on line {line}")
        if not 1 <= second <= 8:
            raise ValueError(f"bad Power second area on line {line}")
        day = None
        if dated:
            day = date.fromisoformat(row["draw_date"])
            if row["draw_date"] != day.isoformat() or day.year - 1911 != int(identifier[:3]):
                raise ValueError(f"invalid ISO/ROC draw date at line {line}")
            if last_date and day <= last_date:
                raise ValueError(f"draw dates not increasing at line {line}")
            last_date = day
        result.append((identifier, nums, second, day.isoformat() if day else None))
    return result

def download_artifact(token):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/artifacts/{EXPECTED_ARTIFACT_ID}/zip",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "OF-LGBM-staging-verifier"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("artifact ZIP unexpectedly large")
    return data

def verify(artifact_zip, archive, output_dir):
    if digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("canonical archive SHA-256 mismatch; do not import")
    if output_dir.resolve() == archive.resolve() or archive.resolve() in output_dir.resolve().parents:
        raise ValueError("staging directory cannot overwrite canonical archive")
    if output_dir.exists():
        raise ValueError("staging directory exists; refusing overwrite")
    with zipfile.ZipFile(io.BytesIO(artifact_zip)) as z:
        names = set(z.namelist())
        expected = {"power-history-dated-candidate.csv",
                    "power-history-dated-research.sqlite", "DATE_AUDIT.json"}
        if not expected.issubset(names):
            raise ValueError("dated stage artifact incomplete")
        for name in expected:
            if z.getinfo(name).file_size > 2_000_000:
                raise ValueError("unusually large artifact member: " + name)
        candidate = z.read("power-history-dated-candidate.csv")
        raw_db = z.read("power-history-dated-research.sqlite")
        origin = json.loads(z.read("DATE_AUDIT.json"))
    if digest(candidate) != DATED_SHA256 or digest(raw_db) != SQLITE_SHA256:
        raise ValueError("pinned artifact SHA-256 mismatch")
    if origin.get("dated_rows") != EXPECTED_ROWS or origin.get("dated_csv_sha256") != DATED_SHA256:
        raise ValueError("artifact audit contradicts pinned source")
    original = parse(archive.read_bytes(), dated=False)
    recovered = parse(candidate, dated=True)
    if len(original) != EXPECTED_ROWS or len(recovered) != EXPECTED_ROWS:
        raise ValueError("row count mismatch")
    if [(x[0], x[1], x[2]) for x in original] != [(x[0], x[1], x[2]) for x in recovered]:
        raise ValueError("dated rows differ from immutable canonical number rows")
    if recovered[0][3] != "2008-01-24" or recovered[-1][3] != "2026-11-16":
        raise ValueError("pinned date bounds mismatch")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "candidate.sqlite"
        db.write_bytes(raw_db)
        with sqlite3.connect(db) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            sql_rows = conn.execute(
                "SELECT draw_id, draw_date, first, second FROM draws ORDER BY draw_id"
            ).fetchall()
    if len(sql_rows) != EXPECTED_ROWS:
        raise ValueError("SQLite row count mismatch")
    for number, (csv_row, sqlite_row) in enumerate(zip(recovered, sql_rows), 1):
        draw_id, draw_date, first, second = sqlite_row
        numbers = tuple(sorted(int(n) for n in first.replace(",", " ").split()))
        if csv_row != (draw_id, numbers, int(second), draw_date):
            raise ValueError(f"SQLite and dated CSV mismatch on row {number}")
    report = {
        "status": "STAGED_REVIEW_ONLY_NOT_OFFICIAL_CERTIFICATION",
        "source_artifact_id": EXPECTED_ARTIFACT_ID,
        "rows_verified": EXPECTED_ROWS,
        "missing_dates": 0,
        "first_draw_id": recovered[0][0],
        "last_draw_id": recovered[-1][0],
        "date_start": recovered[0][3],
        "date_end": recovered[-1][3],
        "csv_sha256": DATED_SHA256,
        "sqlite_sha256": SQLITE_SHA256,
        "canonical_sha256": ARCHIVE_SHA256,
        "sqlite_integrity": "ok",
        "canonical_numbers_preserved": True,
        "official_provenance_independently_authenticated": False,
        "production_import_completed": False,
        "note": "Only checks pinned staging artifact against repository snapshot; never promotes official status."
    }
    output_dir.mkdir(parents=True)
    (output_dir / "power_history_dated_STAGING.csv").write_bytes(candidate)
    (output_dir / "power_history_dated_STAGING.sqlite").write_bytes(raw_db)
    (output_dir / "STAGING_VERIFICATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path,
                   default=Path("verification/of_lgbm_data/power_history_archive.csv"))
    p.add_argument("--artifact-zip", type=Path,
                   help="Optional pre-downloaded GitHub artifact ZIP")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.artifact_zip:
        raw = args.artifact_zip.read_bytes()
    else:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN is required to read workflow artifacts")
        raw = download_artifact(token)
    print(json.dumps(verify(raw, args.archive, args.output_dir), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
