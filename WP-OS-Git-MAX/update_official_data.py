#!/usr/bin/env python3
"""Download Taiwan Lottery SuperLotto638 draws with strict safety checks.
Only trusted validated changes are written. No external dependencies.
"""
import argparse
import csv
import datetime as dt
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "historical_latest_10.csv"
LATEST20 = ROOT / "historical_latest_20.csv"
HISTORY = ROOT / "historical_archive.csv"
URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/SuperLotto638Result"

def parse_response(obj):
    content = obj.get("content")
    if not isinstance(content, dict) or not isinstance(content.get("superLotto638Res"), list):
        raise ValueError("Official API content/superLotto638Res unavailable or changed")
    rows = []
    for item in content["superLotto638Res"]:
        period = str(item.get("period", "")).strip()
        numbers = item.get("drawNumberSize")
        if not (period.isdigit() and len(period) == 9 and isinstance(numbers, list) and len(numbers) == 7):
            raise ValueError("Unexpected draw structure; refusing update")
        first = sorted(int(n) for n in numbers[:6])
        second = int(numbers[6])
        if len(set(first)) != 6 or not all(1 <= n <= 38 for n in first) or not 1 <= second <= 8:
            raise ValueError("Invalid lottery numbers; refusing update")
        rows.append((period, first, second))
    if not rows:
        raise ValueError("No official draws returned")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Duplicate official periods")
    return rows

def fetch(month):
    query = urlencode({"period": "", "month": month, "pageNum": 1, "pageSize": 50})
    request = Request(URL + "?" + query, headers={"User-Agent": "WP-OS-Git-MAX/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=25) as response:
        data = json.load(response)
    return parse_response(data)

def existing(path=None):
    path = DATA if path is None else path
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    out = []
    for r in records:
        first = sorted(int(v) for v in r["first"].split())
        second = int(r["second"])
        if len(first) != 6 or len(set(first)) != 6 or not all(1 <= n <= 38 for n in first) or not 1 <= second <= 8:
            raise ValueError("Existing CSV invalid")
        out.append((r["draw_id"], first, second))
    if len({r[0] for r in out}) != len(out):
        raise ValueError("Existing CSV duplicate periods")
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="YYYY-MM; defaults to current Taiwan month")
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    month = args.month or now.strftime("%Y-%m")
    if not (len(month) == 7 and month[4] == "-" and month[:4].isdigit() and month[5:].isdigit() and 1 <= int(month[5:]) <= 12):
        raise ValueError("Invalid month")
    retrieved = fetch(month)
    old = existing()
    archive = existing(HISTORY)
    by_period = {row[0]: row for row in old}
    for row in archive + retrieved:
        if row[0] in by_period and by_period[row[0]] != row:
            raise ValueError(f"Conflicting official data for {row[0]}; manual review required")
        by_period[row[0]] = row
    merged = sorted(by_period.values(), key=lambda row: int(row[0]))
    # Preserve all verified records for walk-forward evaluation.
    # Accumulate all verified draws in the legacy latest_20 file for compatibility.
    # Selection remains restricted to the separate latest_10 file.
    last20 = merged
    # Keep the legacy ten-draw file for existing snapshots and clients.
    last10 = merged[-10:]
    if len(last10) < 10 or int(last10[-1][0]) < int(old[-1][0] if old else "0"):
        raise ValueError("Insufficient or stale API data; refusing update")
    archive_buffer = io.StringIO(newline="")
    archive_writer = csv.writer(archive_buffer, lineterminator="\n")
    archive_writer.writerow(["draw_id", "first", "second"])
    for draw_id, first, second in merged:
        archive_writer.writerow([draw_id, " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    HISTORY.write_text(archive_buffer.getvalue(), encoding="utf-8")
    latest_buffer = io.StringIO(newline="")
    latest_writer = csv.writer(latest_buffer, lineterminator="\n")
    latest_writer.writerow(["draw_id", "first", "second"])
    for draw_id, first, second in last20:
        latest_writer.writerow([draw_id, " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    LATEST20.write_text(latest_buffer.getvalue(), encoding="utf-8")
    print(f"Stored {len(last20)} draws in {LATEST20.name}; selection windows remain 5/10")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["draw_id", "first", "second"])
    for draw_id, first, second in last10:
        writer.writerow([draw_id, " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    updated = buffer.getvalue()
    previous = DATA.read_text(encoding="utf-8-sig") if DATA.exists() else ""
    if updated != previous:
        DATA.write_text(updated, encoding="utf-8")
        print(f"Updated through draw {last10[-1][0]}")
    else:
        print(f"No changes; through draw {last10[-1][0]}")
if __name__ == "__main__":
    main()

