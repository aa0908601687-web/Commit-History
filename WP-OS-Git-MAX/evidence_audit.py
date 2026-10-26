#!/usr/bin/env python3
"""Independent evidence tools; never modify WP-OS 5/10 scoring."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from wp_os_git_max import load, predict

def audit_history(path):
    draws = load(path)
    if not draws:
        raise ValueError("Empty history")
    ids = [int(d["draw_id"]) for d in draws]
    if any(b <= a for a,b in zip(ids,ids[1:])):
        raise ValueError("Draw periods not increasing")
    return {"count":len(draws),"first_draw":draws[0]["draw_id"],
            "through_draw":draws[-1]["draw_id"],
            "source_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "selection_feature_window":"last 5/10 only"}

def walkforward(draws, count=None):
    if len(draws)<11:
        raise ValueError("Need at least eleven draws")
    ends=list(range(10,len(draws)))
    if count is not None:
        if count<1: raise ValueError("Invalid backtest length")
        ends=ends[-count:]
    hits=Counter()
    second=0
    observations=[]
    for i in ends:
        p=predict(draws[:i])
        a=draws[i]
        k=len(set(p["first"]) & set(a["first"]))
        z=int(p["second"]==a["second"])
        hits[k]+=1
        second+=z
        observations.append({"target_draw":a["draw_id"],"through_draw":p["through_draw"],
                             "first_hits":k,"second_hit":bool(z),
                             "first":p["first"],"second":p["second"],
                             "prediction_sha256":p["sha256"]})
    total=len(ends)
    probability_3plus=sum(math.comb(6,k)*math.comb(32,6-k) for k in range(3,7))/math.comb(38,6)
    return {"tested":total,"first_zone_hit_distribution":{str(i):hits[i] for i in range(7)},
            "first_zone_3plus":sum(hits[i] for i in range(3,7)),
            "first_zone_3plus_rate":sum(hits[i] for i in range(3,7))/total,
            "fair_random_3plus_rate":probability_3plus,
            "second_zone_hits":second,"second_zone_hit_rate":second/total,
            "fair_random_second_hit_rate":1/8,
            "observations":observations,
            "warning":"Historical walk-forward is not a timestamped prospective forecast."}

def verify_forecast_ledger(path, draws):
    """Evaluate immutable externally committed prediction rows; no retrospective relabeling."""
    if not path:
        return {"status":"not_supplied","matched":0,"note":"No pre-draw ledger; historical backtest must not be labelled prospective"}
    actual={d["draw_id"]:d for d in draws}
    with open(path,newline="",encoding="utf-8-sig") as f:
        records=list(csv.DictReader(f))
    required={"target_draw","through_draw","first","second","prediction_sha256"}
    if records and not required.issubset(records[0]):
        raise ValueError("Forecast ledger missing required columns")
    seen=set()
    matched=[]
    for row in records:
        target=row["target_draw"]
        if target in seen: raise ValueError("Duplicate target forecasts")
        seen.add(target)
        first=sorted(int(n) for n in row["first"].replace(","," ").split())
        second=int(row["second"])
        if len(first)!=6 or len(set(first))!=6 or not all(1<=n<=38 for n in first) or not 1<=second<=8:
            raise ValueError("Invalid forecast numbers")
        if not (target.isdigit() and row["through_draw"].isdigit() and int(row["through_draw"])<int(target)):
            raise ValueError("Invalid prediction ordering")
        if not row["prediction_sha256"]:raise ValueError("Missing forecast digest")
        if target in actual:
            a=actual[target]
            matched.append({"target_draw":target,"first_hits":len(set(first)&set(a["first"])),
                            "second_hit":second==a["second"]})
    return {"status":"evaluated","ledger_rows":len(records),"matched":len(matched),
            "results":matched,"warning":"CSV and hashes do not by themselves establish pre-draw timestamps"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--backtest-count",type=int,default=100)
    ap.add_argument("--ledger")
    ap.add_argument("--output")
    args=ap.parse_args()
    report={"history":audit_history(args.csv)}
    draws=load(args.csv)
    report["walkforward"]=walkforward(draws,args.backtest_count) if len(draws)>10 else {"status":"insufficient_history","available_targets":0}
    report["prospective_ledger"]=verify_forecast_ledger(args.ledger,draws)
    report["validation"]={"requested_backtest_targets":args.backtest_count,
                          "available_backtest_targets":max(0,len(draws)-10),
                          "full_requested_backtest":len(draws)-10>=args.backtest_count}
    serialized=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.output: Path(args.output).write_text(serialized,encoding="utf-8")
    print(serialized,end="")
if __name__=="__main__":
    main()
