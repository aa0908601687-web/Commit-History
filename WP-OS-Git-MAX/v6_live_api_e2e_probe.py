#!/usr/bin/env python3
"""Read-only official-API -> three-model -> local QAOA integration probe.

It never changes repository CSV, model weights or production predictions.
Live API success is reported only after a verified HTTP response and full bridge run.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import tempfile
from pathlib import Path

import update_official_data as official
from v6_three_model_qaoa_bridge import run as bridge_run


def previous_month(month):
    year, mon = map(int, month.split("-"))
    return f"{year - (mon == 1):04d}-{12 if mon == 1 else mon - 1:02d}"


def probe(month=None, archive_path=None, fetcher=None, bridge=None, merged_output=None):
    if month is None:
        month = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m")
    dt.date.fromisoformat(month + "-01")  # Reject malformed month before any requests.
    fetcher = fetcher or official.fetch
    bridge = bridge or bridge_run
    archive_path = Path(archive_path or official.HISTORY)
    stored = official.existing(archive_path)
    if len(stored) < 20:
        raise ValueError("Stored archive has fewer than 20 completed draws")
    # Request both months to bridge the month boundary. Never silently accept API errors.
    months = (previous_month(month), month)
    remote = []
    empty_months = []
    for m in months:
        try:
            month_records = fetcher(m)
        except ValueError as exc:
            if m == month and str(exc) == "No official draws returned":
                empty_months.append(m)
                month_records = []
            else:
                raise
        remote.extend(month_records)
    if not remote:
        raise ValueError("Official API returned no validated records")
    incoming = {}
    for draw_id, first, second in remote:
        item = (draw_id, sorted(first), second)
        if draw_id in incoming and incoming[draw_id] != item:
            raise ValueError("Contradictory API records: " + draw_id)
        incoming[draw_id] = item
    by_id = {record[0]: record for record in stored}
    matching = 0
    for draw_id, record in incoming.items():
        if draw_id in by_id:
            if by_id[draw_id] != record:
                raise ValueError("API/archive conflict at " + draw_id)
            matching += 1
        by_id[draw_id] = record
    if matching == 0:
        raise ValueError("No shared official period with repository archive; provenance unverified")
    merged = sorted(by_id.values(), key=lambda item: int(item[0]))
    # merged includes every stored draw; its latest period cannot precede the archive cutoff.
    # Run only on a disposable CSV. Never overwrite the repository archive.
    with tempfile.TemporaryDirectory(prefix="v6-live-probe-") as folder:
        path = Path(folder) / "verified_merge.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("draw_id", "first", "second"))
            for draw_id, first, second in merged:
                writer.writerow((draw_id, " ".join(f"{n:02d}" for n in first), f"{second:02d}"))
        result = bridge(path)
        if result["cutoff"] != merged[-1][0]:
            raise ValueError("Bridge output cutoff differs from validated input")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if merged_output is not None:
            destination = Path(merged_output)
            destination.write_bytes(path.read_bytes())
        if result["source_archive_sha256"] != digest:
            raise ValueError("Bridge input hash mismatch")
    new_periods = [draw_id for draw_id in incoming if int(draw_id) > int(stored[-1][0])]
    return {
        "status": "new_draws_imported_and_verified" if new_periods else "no_new_draws_live_bridge_verified",
        "new_draw_count": len(new_periods),
        "new_draw_periods": sorted(new_periods),
        "latest_draw_advanced": bool(new_periods),
        "months_requested": months,
        "months_without_published_draws": empty_months,
        "api_rows": len(remote),
        "api_distinct_periods": len(incoming),
        "overlap_verified": matching,
        "stored_cutoff": stored[-1][0],
        "merged_cutoff": merged[-1][0],
        "merged_rows": len(merged),
        "verified_merged_sha256": digest,
        "three_model_qaoa_comparison": result,
        "repository_archive_modified": False,
        "trained_ai_verified": False,
        "predictive_advantage_verified": False,
        "production_main_deployed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="Target month in YYYY-MM; default Taiwan current month")
    parser.add_argument("--output", help="JSON output artifact")
    parser.add_argument("--merged-csv-output", help="Verified live-merged CSV for downstream V6 E2E; does not modify archive")
    args = parser.parse_args()
    result = probe(month=args.month, merged_output=args.merged_csv_output)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
