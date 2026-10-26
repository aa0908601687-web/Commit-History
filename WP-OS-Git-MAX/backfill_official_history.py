#!/usr/bin/env python3
"""Backfill WP-OS from official Taiwan Lottery monthly SuperLotto638 API.

Atomic policy: any missing/invalid month or conflict aborts before writing CSVs.
Run with Python 3.11+ from anywhere within the repository.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from update_official_data import fetch, existing

def months():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    for year in range(2008, now.year + 1):
        for month in range(1, 13):
            if (year, month) > (now.year, now.month):
                break
            yield f"{year:04d}-{month:02d}"

def render(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["draw_id", "first", "second"])
    for period, first, second in rows:
        writer.writerow([period, " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    return stream.getvalue()

def main():
    archive = ROOT / "historical_archive.csv"
    original = existing(archive)
    saved = {r[0]: r for r in original}
    official = {}
    scanned = 0
    for month in months():
        err = None
        for attempt in range(4):
            try:
                rows = fetch(month)
                break
            except Exception as exc:
                err = exc
                if attempt != 3:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Official month {month} inaccessible: {err}. No data changed.")
        for record in rows:
            key = record[0]
            if key in official and official[key] != record:
                raise ValueError(f"Conflicting official draw {key}")
            official[key] = record
        scanned += 1
        time.sleep(0.1)
    if len(official) < 1500:
        raise RuntimeError(f"Suspiciously incomplete official archive: {len(official)} draws; no update.")
    for key, row in saved.items():
        if key not in official or official[key] != row:
            raise ValueError(f"Existing draw {key} not independently confirmed; no data changed.")
    full = sorted(official.values(), key=lambda r: int(r[0]))
    if any(int(full[i][0]) >= int(full[i+1][0]) for i in range(len(full)-1)):
        raise RuntimeError("Invalid order")
    payload = render(full)
    # Stage all outputs before replacing any path.
    paths = {
        ROOT / "historical_archive.csv": payload,
        ROOT / "historical_latest_20.csv": payload,
        ROOT / "historical_latest_10.csv": render(full[-10:]),
    }
    for path, data in paths.items():
        tmp = path.with_suffix(".csv.pending")
        tmp.write_text(data, encoding="utf-8")
    for path in paths:
        path.with_suffix(".csv.pending").replace(path)
    manifest = {
        "source": "Taiwan Lottery official SuperLotto638 monthly API",
        "months_queried": scanned,
        "draw_count": len(full),
        "first_draw": full[0][0],
        "last_draw": full[-1][0],
        "archive_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "feature_window": "last 5/10 draws only",
    }
    (ROOT / "history_backfill_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__":
    main()
