#!/usr/bin/env python3
"""Audit Power Lotto dates against original annual Taiwan Lottery ZIPs.

Never invent dates; never update canonical data here. Annual ZIP coverage may
lag current-month API results. A structural pass does not attest provenance
unless original ZIP origin is independently reviewed.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pathlib import Path

def parse_date(value):
    value = value.strip().split("T")[0].split(" ")[0]
    value = re.sub(r"[年月]", "/", value).replace("日", "")
    parts = re.split(r"[-/.]", value)
    if len(parts) != 3 or any(not p.isdigit() for p in parts):
        raise ValueError("unrecognized official draw date: " + repr(value))
    year, month, day = map(int, parts)
    if year <= 1911:
        year += 1911
    return date(year, month, day).isoformat()

def decode(data):
    for encoding in ("utf-8-sig", "cp950", "big5", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("unsupported official archive encoding")

def extract_year(year, zip_path):
    source = Path(zip_path).read_bytes()
    dates = {}
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        for name in z.namelist():
            if name.endswith("/") or not name.lower().endswith(".csv"):
                continue
            rows = csv.reader(io.StringIO(decode(z.read(name))))
            header = next(rows, None)
            if not header:
                continue
            cols = [c.strip().lstrip("\ufeff") for c in header]
            if "遊戲名稱" not in cols or "期別" not in cols:
                continue
            g, p = cols.index("遊戲名稱"), cols.index("期別")
            d = cols.index("開獎日期") if "開獎日期" in cols else None
            for row in rows:
                if len(row) <= g or row[g].strip() != "威力彩":
                    continue
                if d is None or len(row) <= max(p, d):
                    raise ValueError(f"{year}: missing official draw-date field")
                draw_id = row[p].strip()
                if draw_id.isdigit() and len(draw_id) == 8:
                    draw_id = draw_id.zfill(9)
                if len(draw_id) != 9 or not draw_id.isdigit():
                    raise ValueError(f"{year}: invalid draw ID {draw_id!r}")
                draw_date = parse_date(row[d])
                if int(draw_id[:3]) != int(draw_date[:4]) - 1911:
                    raise ValueError(f"{draw_id}: ROC draw/date mismatch")
                if draw_id in dates and dates[draw_id] != draw_date:
                    raise ValueError(f"{draw_id}: inconsistent official dates")
                dates[draw_id] = draw_date
    if not dates:
        raise ValueError(f"{year}: no dated Power Lotto rows found")
    return dates, hashlib.sha256(source).hexdigest()

def fetch_live_dates(month, expected):
    endpoint = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/SuperLotto638Result"
    params = urlencode({"period":"","month":month,"pageNum":1,"pageSize":50})
    url = endpoint + "?" + params
    request = Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    with urlopen(request,timeout=30) as response:
        raw=response.read()
    payload=json.loads(raw)
    entries=payload.get("content",{}).get("superLotto638Res")
    if not isinstance(entries,list) or not entries:
        raise ValueError("official live API returned no draws")
    found={}
    for entry in entries:
        draw_id=str(entry.get("period","")).strip()
        if draw_id not in expected:
            continue
        nums=entry.get("drawNumberSize")
        if not isinstance(nums,list) or len(nums)!=7:
            raise ValueError(f"{draw_id}: malformed official API numbers")
        actual=(tuple(sorted(int(n) for n in nums[:6])),int(nums[6]))
        saved=expected[draw_id]
        canonical=(tuple(sorted(int(n) for n in saved["first"].split())),int(saved["second"]))
        if actual!=canonical:
            raise ValueError(f"{draw_id}: official API numbers conflict with canonical record")
        draw_date=parse_date(str(entry.get("lotteryDate","")))
        if int(draw_id[:3])!=int(draw_date[:4])-1911:
            raise ValueError(f"{draw_id}: official API date/year mismatch")
        if draw_id in found and found[draw_id]!=draw_date:
            raise ValueError(f"{draw_id}: conflicting official live dates")
        found[draw_id]=draw_date
    return found,{"month":month,"url":url,
                  "response_sha256":hashlib.sha256(raw).hexdigest(),
                  "matched_draws":len(found)}

def audit(archive_dir, current_archive, output):
    archive_dir = Path(archive_dir)
    with Path(current_archive).open(encoding="utf-8-sig", newline="") as fp:
        records = list(csv.DictReader(fp))
    expected = {}
    for row in records:
        draw_id = row["draw_id"]
        if draw_id in expected:
            raise ValueError("duplicate canonical draw ID " + draw_id)
        expected[draw_id] = row
    dates = {}
    sources = []
    for year in range(2008, 2027):
        p = archive_dir / f"{year}.zip"
        if not p.is_file():
            raise FileNotFoundError(f"missing official ZIP {p}")
        result, sha = extract_year(year, p)
        for draw_id, dt in result.items():
            if draw_id in dates and dates[draw_id] != dt:
                raise ValueError(f"cross-year date conflict {draw_id}")
            dates[draw_id] = dt
        sources.append({"year":year,"archive_sha256":sha,"dated_power_rows":len(result)})
    # Monthly official annual ZIPs can lag recent draws. Require both official
    # API number zones to match before accepting a live draw date.
    missing_before_live = sorted(set(expected) - set(dates), key=int)
    live_source = None
    if missing_before_live:
        taiwan_now = datetime.now(timezone(timedelta(hours=8)))
        month = taiwan_now.strftime("%Y-%m")
        live, live_source = fetch_live_dates(month, expected)
        for draw_id, draw_date in live.items():
            if draw_id in dates and dates[draw_id] != draw_date:
                raise ValueError(f"{draw_id}: official ZIP / API date conflict")
            dates[draw_id] = draw_date
    ordered = sorted(expected, key=int)
    last_date = None
    for draw_id in ordered:
        if draw_id not in dates:
            continue
        current = dates[draw_id]
        if last_date is not None and current <= last_date:
            raise ValueError(f"non-increasing official dates at {draw_id}")
        last_date = current
    missing = [draw_id for draw_id in ordered if draw_id not in dates]
    report = {
        "kind":"official_yearly_zip_date_coverage_audit",
        "source_archive_years":19,
        "canonical_rows":len(expected),
        "dated_matching_ids":len(expected)-len(missing),
        "missing_date_ids":missing,
        "first_draw_id":ordered[0],
        "last_draw_id":ordered[-1],
        "annual_sources":sources,
        "live_api_source":live_source,
        "missing_before_live_api":missing_before_live,
        "all_canonical_dates_present":not missing,
        "provenance_independently_authenticated":False,
        "of_lgbm_production_import_completed":False,
        "note":"Original downloaded ZIPs checked structurally; annual ZIPs may lag live draws. No dates were inferred or production data overwritten."
    }
    # A complete matching set is staged only as an evidence artifact, never
    # copied over the canonical OF-LGBM source or its pinned manifest.
    if not missing:
        candidate = Path(output).with_name("power-official-dated-candidate.csv")
        if candidate.exists():
            raise ValueError("candidate output exists; refusing overwrite")
        with candidate.open("x",encoding="utf-8",newline="") as fp:
            writer=csv.writer(fp,lineterminator="\n")
            writer.writerow(["draw_id","draw_date","first","second"])
            for draw_id in ordered:
                row=expected[draw_id]
                writer.writerow([draw_id,dates[draw_id],row["first"],row["second"]])
        report["dated_candidate_path"]=str(candidate)
        report["dated_candidate_sha256"]=hashlib.sha256(candidate.read_bytes()).hexdigest()
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir",required=True)
    p.add_argument("--current-archive",required=True)
    p.add_argument("--output",required=True)
    a = p.parse_args()
    result = audit(a.archive_dir,a.current_archive,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!="annual_sources"},ensure_ascii=False))
