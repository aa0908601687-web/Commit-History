#!/usr/bin/env python3
"""Backfill WP-OS from official Taiwan Lottery annual archives plus current month API.

Atomic policy: incomplete, invalid, or conflicting official data aborts before
any repository CSV is replaced. Selection features remain limited to 5/10 draws.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from update_official_data import fetch, existing

DOWNLOAD_API = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/ResultDownload"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/141 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.taiwanlottery.com/lotto/history/result_download/",
    "Origin": "https://www.taiwanlottery.com",
}
FIRST_YEAR = 2008


def request_bytes(url, accept="application/octet-stream", timeout=60):
    headers = dict(HEADERS)
    headers["Accept"] = accept
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        return response.read()


def annual_download(year):
    url = DOWNLOAD_API + "?" + urlencode({"year": year})
    obj = json.loads(request_bytes(url, "application/json", 30).decode("utf-8"))
    content = obj.get("content")
    if obj.get("rtCode") != 0 or not isinstance(content, dict):
        raise ValueError(f"Official ResultDownload response invalid for {year}")
    path = content.get("path")
    if not isinstance(path, str) or not path.lower().startswith("https://"):
        raise ValueError(f"Official annual archive path missing for {year}")
    return path, request_bytes(path)


def decode_csv(raw):
    for encoding in ("utf-8-sig", "cp950", "big5", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeError("Official CSV encoding unsupported")


def parse_annual_zip(year, payload):
    rows = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = [name for name in archive.namelist() if not name.endswith("/")]
        if not files:
            raise ValueError(f"Official {year} ZIP is empty")
        for name in files:
            text = decode_csv(archive.read(name))
            reader = csv.reader(io.StringIO(text))
            header = next(reader, None)
            if not header:
                continue
            normalized = [cell.strip().lstrip("\ufeff") for cell in header]
            try:
                game_i = normalized.index("遊戲名稱")
                period_i = normalized.index("期別")
                number_i = [normalized.index(f"獎號{i}") for i in range(1, 7)]
            except ValueError:
                continue
            for line in reader:
                if len(line) <= max(period_i, *number_i):
                    continue
                if line[game_i].strip() != "威力彩":
                    continue
                period = line[period_i].strip()
                if period.isdigit() and len(period) == 8:
                    period = period.zfill(9)
                first = sorted(int(line[i].strip()) for i in number_i)
                # The official common schema stores the second-zone number
                # immediately after the six first-zone numbers.
                second_i = number_i[-1] + 1
                if len(line) <= second_i or not line[second_i].strip():
                    raise ValueError(f"Official {year} 威力彩 row lacks second-zone number: {period}")
                second = int(line[second_i].strip())
                if not (period.isdigit() and len(period) == 9):
                    raise ValueError(f"Invalid official period in {year}: {period!r}")
                if len(set(first)) != 6 or not all(1 <= n <= 38 for n in first):
                    raise ValueError(f"Invalid first-zone numbers for {period}")
                if not 1 <= second <= 8:
                    raise ValueError(f"Invalid second-zone number for {period}")
                rows.append((period, first, second))
    if not rows:
        raise ValueError(f"No 威力彩 records found in official {year} archive")
    unique = {}
    for row in rows:
        if row[0] in unique and unique[row[0]] != row:
            raise ValueError(f"Conflicting official annual draw {row[0]}")
        unique[row[0]] = row
    return list(unique.values())


def render(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["draw_id", "first", "second"])
    for period, first, second in rows:
        writer.writerow([period, " ".join(f"{n:02d}" for n in first), f"{second:02d}"])
    return stream.getvalue()


def main():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    archive = ROOT / "historical_archive.csv"
    saved = {row[0]: row for row in existing(archive)}
    official = {}
    sources = []

    for year in range(FIRST_YEAR, now.year + 1):
        err = None
        for attempt in range(4):
            try:
                path, payload = annual_download(year)
                year_rows = parse_annual_zip(year, payload)
                break
            except Exception as exc:
                err = exc
                if attempt != 3:
                    time.sleep((5, 10, 20)[attempt])
        else:
            raise RuntimeError(f"Official annual archive {year} inaccessible: {err}. No data changed.")
        for row in year_rows:
            if row[0] in official and official[row[0]] != row:
                raise ValueError(f"Conflicting official draw {row[0]}")
            official[row[0]] = row
        sources.append({"year": year, "path": path, "draws": len(year_rows)})
        time.sleep(2)

    # Annual files update monthly. Add the live current-month endpoint so the
    # archive also contains draws after the latest annual-file refresh.
    month = now.strftime("%Y-%m")
    try:
        live_rows = fetch(month)
    except Exception as exc:
        raise RuntimeError(f"Official current month {month} inaccessible: {exc}. No data changed.")
    for row in live_rows:
        if row[0] in official and official[row[0]] != row:
            raise ValueError(f"Annual/live conflict for official draw {row[0]}")
        official[row[0]] = row

    if len(official) < 1500:
        raise RuntimeError(f"Suspiciously incomplete official archive: {len(official)} draws; no update.")
    for key, row in saved.items():
        if key not in official or official[key] != row:
            raise ValueError(f"Existing draw {key} not independently confirmed; no data changed.")

    full = sorted(official.values(), key=lambda row: int(row[0]))
    if any(int(full[i][0]) >= int(full[i + 1][0]) for i in range(len(full) - 1)):
        raise RuntimeError("Invalid official period order")

    payload = render(full)
    paths = {
        ROOT / "historical_archive.csv": payload,
        # Compatibility file remains cumulative; model features still read
        # only the dedicated ten-draw snapshot.
        ROOT / "historical_latest_20.csv": payload,
        ROOT / "historical_latest_10.csv": render(full[-10:]),
    }
    for path, data in paths.items():
        path.with_suffix(".csv.pending").write_text(data, encoding="utf-8")
    for path in paths:
        path.with_suffix(".csv.pending").replace(path)

    manifest = {
        "source": "Taiwan Lottery official annual ZIP archives + current-month API",
        "annual_archives": sources,
        "live_month": month,
        "draw_count": len(full),
        "first_draw": full[0][0],
        "last_draw": full[-1][0],
        "archive_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "feature_window": "last 5/10 draws only",
        "backtest_window": 15,
    }
    (ROOT / "history_backfill_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "annual_archives"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
