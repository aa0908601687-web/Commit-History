#!/usr/bin/env python3
"""OF-LGBM-only chronological CSV audit, not official-record certification."""
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

RULES={"power":(38,6,"second"),"lotto":(49,6,"special"),"539":(39,5,None)}

def audit_csv(filename,game,min_rows=17):
    if game not in RULES: raise ValueError("unsupported game")
    maximum,count,extra=RULES[game]
    p=Path(filename)
    previous_year=previous_seq=previous_date=None
    ids=set()
    dates=[]
    with p.open(encoding="utf-8-sig",newline="") as source:
        reader=csv.DictReader(source)
        required={"draw_id","first"}|({extra} if extra else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("missing mandatory data columns")
        has_date="draw_date" in reader.fieldnames
        if game in ("lotto","539") and not has_date:
            raise ValueError("real-data mirror requires draw_date")
        for line,row in enumerate(reader,2):
            try:
                identifier=int(row["draw_id"])
                numbers=[int(v) for v in row["first"].replace(","," ").split()]
                extra_value=int(row[extra]) if extra else None
                this_date=date.fromisoformat(row["draw_date"]) if has_date else None
            except (KeyError,ValueError,TypeError,AttributeError) as err:
                raise ValueError(f"invalid values on line {line}") from err
            if len(numbers)!=count or len(set(numbers))!=count or any(n<1 or n>maximum for n in numbers):
                raise ValueError(f"invalid first-zone numbers on line {line}")
            if extra=="second" and not 1<=extra_value<=8:
                raise ValueError(f"invalid Power second-zone number on line {line}")
            if extra=="special" and (not 1<=extra_value<=maximum or extra_value in numbers):
                raise ValueError(f"invalid Lotto special number on line {line}")
            year,seq=divmod(identifier,1000000)
            if year<1 or seq<1 or identifier in ids:
                raise ValueError(f"duplicate or invalid identifier on line {line}")
            if previous_year is not None:
                if year==previous_year and seq!=previous_seq+1:
                    raise ValueError(f"missing or nonconsecutive draw ID on line {line}")
                if year!=previous_year and (year!=previous_year+1 or seq!=1):
                    raise ValueError(f"invalid ROC year rollover on line {line}")
            if this_date is not None:
                if row["draw_date"]!=this_date.isoformat():
                    raise ValueError(f"nonstandard ISO date on line {line}")
                if this_date.year-1911!=year:
                    raise ValueError(f"ROC draw ID/year mismatch on line {line}")
                if previous_date is not None and this_date<=previous_date:
                    raise ValueError(f"nonincreasing draw date on line {line}")
                previous_date=this_date
            ids.add(identifier)
            dates.append((str(identifier),this_date.isoformat() if this_date else None))
            previous_year,previous_seq=year,seq
    if len(dates)<min_rows: raise ValueError("not enough historical draws")
    return {"game":game,"rows":len(dates),
            "date_coverage":sum(d is not None for _,d in dates),
            "date_validation_status":"complete_format_only_not_official" if has_date else "missing_date_column",
            "first_id":dates[0][0],"last_id":dates[-1][0],
            "first_date":dates[0][1],"last_date":dates[-1][1],
            "chronological_and_legal_range_verified":True,
            "csv_dates_consistent":bool(dates[0][1]),
            "record_by_record_official_provenance_verified":False,
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}

def compare_official_normalized(snapshot, official, game):
    """Compare against independently obtained, normalized official CSV; never fill missing values."""
    def read_rows(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"draw_id", "draw_date", "first"} | ({"second"} if game == "power" else {"special"} if game == "lotto" else set())
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("official comparison requires normalized dated CSV columns")
            rows = {}
            for row in reader:
                identifier = row["draw_id"].strip()
                if identifier in rows:
                    raise ValueError("duplicate official draw_id: " + identifier)
                rows[identifier] = row
            return rows
    reference = read_rows(official)
    with Path(snapshot).open(encoding="utf-8-sig", newline="") as source:
        local = list(csv.DictReader(source))
    issues = []
    compared = 0
    for row in local:
        identifier = row["draw_id"].strip()
        expected = reference.get(identifier)
        if expected is None:
            issues.append({"draw_id": identifier, "reason": "official_draw_missing"})
            continue
        compared += 1
        def normalized_numbers(value):
            return sorted(int(v) for v in value.replace(",", " ").split())
        if normalized_numbers(row["first"]) != normalized_numbers(expected["first"]):
            issues.append({"draw_id": identifier, "reason": "first_zone_mismatch"})
        extra = RULES[game][2]
        if extra and int(row[extra]) != int(expected[extra]):
            issues.append({"draw_id": identifier, "reason": "extra_zone_mismatch"})
        if not row.get("draw_date"):
            issues.append({"draw_id": identifier, "reason": "snapshot_date_missing"})
        elif row["draw_date"] != expected["draw_date"]:
            issues.append({"draw_id": identifier, "reason": "date_mismatch"})
    return {"official_comparison_performed": True, "matched_draw_ids": compared,
            "snapshot_rows": len(local), "issues": issues,
            "official_rows_verified": len(issues) == 0 and compared == len(local),
            "official_file_sha256": hashlib.sha256(Path(official).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--game",required=True,choices=tuple(RULES))
    parser.add_argument("--csv",required=True)
    parser.add_argument("--output")
    parser.add_argument("--official-csv", help="Independently sourced normalized official CSV for strict comparison")
    args=parser.parse_args()
    message=json.dumps(audit_csv(args.csv,args.game),indent=2,ensure_ascii=False,sort_keys=True)+"\n"
    if args.output: Path(args.output).write_text(message,encoding="utf-8")
    print(message,end="")

if __name__=="__main__": main()
