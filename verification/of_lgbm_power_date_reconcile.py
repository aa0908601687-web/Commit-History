#!/usr/bin/env python3
"""Fail-closed Power draw-date import from an independently obtained source CSV.

Source must have draw_id,draw_date,first,second. This tool NEVER downloads
or invents dates; it refuses to publish partially matched or conflicting rows.
Use a separately inspected official export adapted to this schema.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

FIELDS = ("draw_id", "draw_date", "first", "second")

def read_rows(path, dated):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = set(FIELDS if dated else ("draw_id", "first", "second"))
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("missing mandatory columns")
        rows = {}
        previous_date = None
        previous_id = None
        for line, row in enumerate(reader, 2):
            ident = row["draw_id"].strip()
            if not ident.isdigit() or len(ident) != 9 or ident in rows:
                raise ValueError(f"duplicate or malformed draw id at line {line}")
            if previous_id is not None and int(ident) <= previous_id:
                raise ValueError(f"nonincreasing draw ID at line {line}")
            previous_id = int(ident)
            try:
                nums = tuple(sorted(int(n) for n in row["first"].replace(",", " ").split()))
                second = int(row["second"])
                d = date.fromisoformat(row["draw_date"]) if dated else None
            except (TypeError, ValueError) as ex:
                raise ValueError(f"invalid record at line {line}") from ex
            if len(nums) != 6 or len(set(nums)) != 6 or any(not 1 <= n <= 38 for n in nums) or not 1 <= second <= 8:
                raise ValueError(f"invalid Power number at line {line}")
            if dated:
                if row["draw_date"] != d.isoformat() or d.year - 1911 != int(ident[:3]):
                    raise ValueError(f"invalid ISO date/ROC year at line {line}")
                if previous_date and d <= previous_date:
                    raise ValueError(f"dates not strictly increasing at line {line}")
                previous_date = d
            rows[ident] = (nums, second, d.isoformat() if d else None)
    if not rows:
        raise ValueError("no rows")
    return rows

def reconcile(archive, source, output, report):
    """Write atomically only after full validation; never overwrite archive."""
    archive, source, output, report = map(Path, (archive, source, output, report))
    if output.resolve() in (archive.resolve(), source.resolve()):
        raise ValueError("output must not overwrite source or archive")
    if output.resolve() == report.resolve() or report.resolve() in (archive.resolve(), source.resolve()):
        raise ValueError("report path must be separate from all inputs and output")
    if output.exists() or report.exists():
        raise ValueError("output or report already exists; refusing overwrite")
    old, dated = read_rows(archive, False), read_rows(source, True)
    if set(old) != set(dated):
        missing = sorted(set(old)-set(dated))
        extra = sorted(set(dated)-set(old))
        raise ValueError(f"source draw IDs differ: missing={len(missing)} extra={len(extra)}")
    for ident, rec in old.items():
        if rec[:2] != dated[ident][:2]:
            raise ValueError(f"drawn numbers differ at {ident}")
    # Date order must also agree with the canonical archive draw order.
    previous_date = None
    for ident in old:
        current_date = dated[ident][2]
        if previous_date is not None and current_date <= previous_date:
            raise ValueError(f"source date order disagrees with archive at {ident}")
        previous_date = current_date
    # Preserve the original row order and numbers verbatim (normalized spaces).
    result_lines = []
    for ident, (first, second, _) in old.items():
        result_lines.append([ident, dated[ident][2], " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    import io
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(result_lines)
    payload = buffer.getvalue().encode("utf-8")
    audit = {
        "rows": len(old), "dated_rows": len(result_lines),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "date_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "first_draw_id": next(iter(old)), "last_draw_id": next(reversed(old)),
        "numbers_preserved": True, "draw_dates_complete": True,
        "date_and_id_order_cross_checked": True,
        "independent_official_provenance_verified": False,
        "note": "Structural reconciliation only. Official authentication requires a documented independently reviewed original official export."
    }
    # Publish both outputs as a unit from the caller's perspective. If either
    # write fails, remove only files created by this invocation. In particular,
    # a failed report write must not leave a seemingly valid dated CSV behind.
    output_created = report_created = False
    try:
        with output.open("xb") as f:
            output_created = True
            f.write(payload)
        with report.open("x", encoding="utf-8") as f:
            report_created = True
            json.dump(audit, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except BaseException:
        if report_created:
            report.unlink(missing_ok=True)
        if output_created:
            output.unlink(missing_ok=True)
        raise
    return audit

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--dated-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    print(json.dumps(reconcile(a.archive, a.dated_source, a.output, a.report),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
