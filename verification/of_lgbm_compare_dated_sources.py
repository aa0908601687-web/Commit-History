#!/usr/bin/env python3
"""Compare archived Power draw dates semantically, never certify or deploy."""
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

def load(path):
    raw=Path(path).read_bytes()
    with Path(path).open(newline="",encoding="utf-8-sig") as handle:
        reader=csv.DictReader(handle)
        required={"draw_id","draw_date","first","second"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing required columns")
        parsed={}
        for line, row in enumerate(reader,2):
            draw_id=row["draw_id"].strip()
            day=date.fromisoformat(row["draw_date"].strip()).isoformat()
            nums=tuple(sorted(int(v) for v in row["first"].replace(","," ").split()))
            second=int(row["second"])
            if len(draw_id)!=9 or not draw_id.isascii() or not draw_id.isdecimal() or draw_id in parsed:
                raise ValueError(f"{path}: invalid or duplicate draw ID at line {line}")
            if len(nums)!=6 or len(set(nums))!=6 or not all(1<=v<=38 for v in nums) or not 1<=second<=8:
                raise ValueError(f"{path}: invalid winning numbers at line {line}")
            if date.fromisoformat(day).year-1911!=int(draw_id[:3]):
                raise ValueError(f"{path}: date/year mismatch at line {line}")
            parsed[draw_id]=(day,nums,second)
    return parsed,hashlib.sha256(raw).hexdigest()

def compare(previous,current,out):
    old,old_sha=load(previous)
    new,new_sha=load(current)
    ids=sorted(set(old)|set(new))
    differences=[{"draw_id":ident,
                  "previous":str(old.get(ident)), "fresh":str(new.get(ident))}
                 for ident in ids if old.get(ident)!=new.get(ident)]
    report={"previous_rows":len(old),"fresh_rows":len(new),
            "previous_sha256":old_sha,"fresh_sha256":new_sha,
            "byte_identical":old_sha==new_sha,
            "semantic_differences":len(differences),
            "differences":differences[:10],
            "official_source_independently_approved":False,
            "production_import_completed":False,
            "note":"This compares two independent workflow outputs, not a production approval."}
    Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="differences"},ensure_ascii=False))
    if len(old)!=1964 or len(new)!=1964 or differences:
        raise ValueError("dated candidate CSV files are not semantically identical")
    return report

if __name__=="__main__":
    if len(sys.argv)!=4: raise SystemExit("usage: compare.py PRIOR.csv FRESH.csv REPORT.json")
    compare(*sys.argv[1:])
