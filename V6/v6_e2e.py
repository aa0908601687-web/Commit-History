"""V6 isolated end-to-end test pipeline using verified-format historical CSV.
Uses trailing 5/10 draws only; no ABE, real QPU, or trained AI.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from v6_lottery_pipeline import pick

def load_history(path):
    draws=[]
    with open(path,encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            first=[int(x) for x in row["first"].replace(","," ").split()]
            second=int(row["second"])
            if len(first)!=6 or len(set(first))!=6 or not all(1<=x<=38 for x in first) or not 1<=second<=8:
                raise ValueError("Invalid lottery draw")
            draws.append({"draw_id":str(row["draw_id"]),"first":first,"second":second})
    if len(draws)<10 or len({x["draw_id"] for x in draws})!=len(draws):
        raise ValueError("Need 10+ unique historical draws")
    return draws

def score(history, zone):
    last5=history[-5:]
    last10=history[-10:]
    n=38 if zone=="first" else 8
    def frequencies(draws):
        counts=[0.0]*n
        for row in draws:
            numbers=row["first"] if zone=="first" else [row["second"]]
            for v in numbers: counts[v-1]+=1/len(draws)
        return counts
    a,b=frequencies(last5),frequencies(last10)
    # Fixed heuristic; no evidence it predicts future draws.
    return [0.6*x+0.4*y for x,y in zip(a,b)]

def predict(history):
    if len(history)<10: raise ValueError("Need 10 prior draws")
    first=score(history,"first")
    second=score(history,"second")
    choice=pick(first,second)
    result={"schema_version":1,"through_draw":history[-1]["draw_id"],
            "feature_windows":[5,10],"feature_weight_5":0.6,
            "feature_weight_10":0.4,"historical_rows":len(history),
            "first":choice["first"],"second":choice["second"],
            "qaoa_shortlist_size":choice["shortlist_size"],
            "mode":"isolated_heuristic_qaoa_research",
            "real_qpu":False,"abe_integration":False,
            "trained_ai_integrated":False,"historical_backtest_performed":False,
            "predictive_advantage_verified":False}
    packed=json.dumps(result,sort_keys=True,separators=(",",":"))
    result["sha256"]=hashlib.sha256(packed.encode()).hexdigest()
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--output")
    args=p.parse_args()
    result=predict(load_history(args.csv))
    payload=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if args.output: Path(args.output).write_text(payload,encoding="utf-8")
    print(payload,end="")

if __name__=="__main__":main()
