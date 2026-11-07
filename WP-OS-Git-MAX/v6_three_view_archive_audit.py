#!/usr/bin/env python3
"""Fail-closed audit of V6 three-view archive coverage."""
import argparse, csv, json, hashlib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--csv",required=True)
p.add_argument("--expected-first-draw",required=True)
p.add_argument("--expected-last-draw",required=True)
p.add_argument("--expected-count",required=True,type=int)
p.add_argument("--output",default="v6-three-view-audit.json")
a=p.parse_args()
with open(a.csv,encoding="utf-8-sig",newline="") as f:
    reader=csv.DictReader(f)
    if reader.fieldnames!=["draw_id","first","second"]:
        raise SystemExit("FAIL: invalid CSV header")
    rows=list(reader)
ids=[r["draw_id"] for r in rows]
if not ids or len(set(ids))!=len(ids) or ids!=sorted(ids):
    raise SystemExit("FAIL: empty, duplicate or unsorted draw IDs")
for r in rows:
    nums=[int(n) for n in r["first"].split()]
    if len(nums)!=6 or len(set(nums))!=6 or any(n<1 or n>38 for n in nums) or not 1<=int(r["second"])<=8:
        raise SystemExit("FAIL: invalid draw numbers")
report={"rows":len(rows),"first_draw":ids[0],"last_draw":ids[-1],
        "expected_count":a.expected_count,"expected_first_draw":a.expected_first_draw,
        "expected_last_draw":a.expected_last_draw,
        "sha256":hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),
        "latest_5_rows":min(5,len(rows)),"latest_10_rows":min(10,len(rows)),
        "archive_coverage_verified":len(rows)==a.expected_count and ids[0]==a.expected_first_draw and ids[-1]==a.expected_last_draw}
Path(a.output).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
if not report["archive_coverage_verified"]:
    raise SystemExit("FAIL: supplied archive does not match independently verified expected boundaries/count")
