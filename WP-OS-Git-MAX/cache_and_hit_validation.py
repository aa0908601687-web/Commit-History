#!/usr/bin/env python3
"""Reproducible lottery research caches and exact hit verification.

The cache is a performance/reproducibility aid, not a predictive advantage.
Prediction snapshots must be published BEFORE the target draw to establish timing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

def load_draws(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            period = str(item["draw_id"])
            first = sorted(int(x) for x in item["first"].replace(",", " ").split())
            second = int(item["second"])
            if not(period.isdigit() and len(period)==9 and len(first)==6 and len(set(first))==6 and all(1<=x<=38 for x in first) and 1<=second<=8):
                raise ValueError("Invalid record: "+period)
            rows.append({"draw_id":period,"first":first,"second":second})
    ids=[x["draw_id"] for x in rows]
    if len(ids)!=len(set(ids)) or ids!=sorted(ids):
        raise ValueError("Duplicate or unsorted periods")
    return rows

def cache_features(draws):
    if len(draws)<10: raise ValueError("Need 10 completed draws")
    history=draws[-10:]
    def freq(data, upper, key):
        return {str(n):sum(n in d[key] if key=="first" else n==d[key] for d in data) for n in range(1,upper+1)}
    out={"through_draw":history[-1]["draw_id"],"source_sha256":digest(history),
         "first_5":freq(history[-5:],38,"first"),"first_10":freq(history,38,"first"),
         "second_5":freq(history[-5:],8,"second"),"second_10":freq(history,8,"second")}
    out["sha256"]=digest(out)
    return out


CACHE_SCHEMA_VERSION = 2
CACHE_MODEL_VERSION = "wp-os-git-max-5-10-v1"

def build_versioned_cache(draws):
    obj = cache_features(draws)
    obj["cache_schema_version"] = CACHE_SCHEMA_VERSION
    obj["model_version"] = CACHE_MODEL_VERSION
    obj["sha256"] = digest({k: v for k, v in obj.items() if k != "sha256"})
    return obj

def get_versioned_cache(draws, path):
    """Reuse a verified cache only for exactly matching input, model and schema."""
    expected = build_versioned_cache(draws)
    path = Path(path)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            checksum = cached.get("sha256")
            intact = isinstance(checksum, str) and checksum == digest(
                {k: v for k, v in cached.items() if k != "sha256"})
            if intact and cached == expected:
                return cached, "hit"
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return expected, "refreshed"

def make_prediction_snapshot(draws, target_draw):
    from wp_os_git_max import predict
    if not (target_draw.isdigit() and len(target_draw)==9 and target_draw>draws[-1]["draw_id"]):
        raise ValueError("Target period must be a later, valid 9-digit draw ID")
    result=predict(draws)
    obj={"target_draw":target_draw,"through_draw":draws[-1]["draw_id"],
         "history_sha256":digest(draws[-10:]),"first":result["first"],"second":result["second"],
         "model_prediction_sha256":result["sha256"],
         "note":"Publish/commit before drawing; local hash alone cannot prove timing"}
    obj["sha256"]=digest(obj)
    return obj

def validate(prediction, draws):
    required=("target_draw","through_draw","history_sha256","first","second","model_prediction_sha256","sha256")
    if any(k not in prediction for k in required):
        raise ValueError("Incomplete snapshot")
    actual_sha=prediction["sha256"]
    if digest({k:v for k,v in prediction.items() if k!="sha256"})!=actual_sha:
        raise ValueError("Prediction snapshot has been altered")
    target=prediction["target_draw"]
    matching=[r for r in draws if r["draw_id"]==target]
    if len(matching)!=1: raise ValueError("Target draw not yet found")
    a=matching[0]
    before=[r for r in draws if r["draw_id"]<=prediction["through_draw"]]
    if len(before)<10 or before[-1]["draw_id"]!=prediction["through_draw"] or digest(before[-10:])!=prediction["history_sha256"]:
        raise ValueError("Snapshot historical data mismatch")
    if target<=prediction["through_draw"]: raise ValueError("Lookahead target")
    f=prediction["first"]
    s=prediction["second"]
    if len(f)!=6 or len(set(f))!=6 or any(not isinstance(n,int) or n<1 or n>38 for n in f) or not isinstance(s,int) or not 1<=s<=8:
        raise ValueError("Invalid prediction")
    hits=len(set(f)&set(a["first"]))
    second_hit=int(s==a["second"])
    return {"target_draw":target,"first_zone_hits":hits,"first_zone_match_fraction":hits/6,
            "second_zone_hit":second_hit,"jackpot_hit":int(hits==6 and second_hit==1)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--feature-cache")
    ap.add_argument("--snapshot")
    ap.add_argument("--target-draw")
    ap.add_argument("--validate-snapshot")
    args=ap.parse_args()
    rows=load_draws(args.csv)
    if args.validate_snapshot:
        snap=json.loads(Path(args.validate_snapshot).read_text(encoding="utf-8"))
        print(json.dumps(validate(snap,rows),indent=2,ensure_ascii=False))
    elif args.snapshot:
        if not args.target_draw: ap.error("--target-draw is required with --snapshot")
        snap=make_prediction_snapshot(rows,args.target_draw)
        output=Path(args.snapshot)
        if output.exists(): raise FileExistsError("Refusing to overwrite immutable prediction snapshot")
        output.write_text(json.dumps(snap,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        print(f"Created prediction snapshot: {output}")
    else:
        if args.feature_cache:
            obj, status = get_versioned_cache(rows, args.feature_cache)
            print("Cache status: " + status)
        else:
            obj = build_versioned_cache(rows)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
if __name__=="__main__":
    main()
