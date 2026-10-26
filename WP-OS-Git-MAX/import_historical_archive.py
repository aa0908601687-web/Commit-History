#!/usr/bin/env python3
"""Safely merge independently verified historical SuperLotto638 CSV into the archive.

Input CSV: draw_id,first,second (official data should first be normalized and checked).
Dry-run is the default. This tool never changes 5/10 feature windows or selection.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "historical_archive.csv"

def load_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames != ["draw_id", "first", "second"]:
            raise ValueError("Expected draw_id,first,second columns")
        rows = {}
        for row in reader:
            period = row["draw_id"].strip()
            first = sorted(int(x) for x in row["first"].replace(",", " ").split())
            second = int(row["second"])
            if (len(period) != 9 or not period.isascii() or not period.isdigit()
                    or len(first) != 6 or len(set(first)) != 6
                    or not all(1 <= x <= 38 for x in first)
                    or not 1 <= second <= 8):
                raise ValueError("Invalid draw: " + period)
            item = (tuple(first), second)
            if period in rows:
                raise ValueError("Duplicate period: " + period)
            rows[period] = item
    return rows

def prepare(existing_path, source_path):
    existing = load_rows(existing_path) if Path(existing_path).exists() else {}
    incoming = load_rows(source_path)
    for period, values in incoming.items():
        if period in existing and existing[period] != values:
            raise ValueError("Conflicting period (no overwrite): " + period)
    merged = dict(existing)
    merged.update(incoming)
    periods = sorted(merged, key=int)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["draw_id", "first", "second"])
    for period in periods:
        first, second = merged[period]
        writer.writerow([period, " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    contents = output.getvalue()
    report = {"previous_count":len(existing),"incoming_count":len(incoming),
              "new_draws":len(merged)-len(existing),"total":len(merged),
              "first_draw":periods[0] if periods else None,
              "last_draw":periods[-1] if periods else None,
              "archive_sha256":hashlib.sha256(contents.encode()).hexdigest()}
    return contents, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Verified normalized history CSV")
    parser.add_argument("--archive", default=str(ARCHIVE))
    parser.add_argument("--write", action="store_true", help="Write validated archive (default dry-run)")
    args = parser.parse_args()
    contents, report = prepare(args.archive, args.source)
    report["wrote"] = bool(args.write)
    if args.write:
        target = Path(args.archive)
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(contents, encoding="utf-8")
        temp.replace(target)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
