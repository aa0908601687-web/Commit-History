"""V6 isolated, deterministic statistical ensemble + local QAOA.
This is NOT pretrained AI or verified predictive modeling.
"""
import argparse
import json
import math
from v6_e2e import load_history,score
from v6_lottery_pipeline import pick

def bayesian_scores(history,zone,alpha=2.0):
    n=38 if zone=="first" else 8
    last=history[-10:]
    baseline=(6/38 if zone=="first" else 1/8)
    counts=[0]*n
    for row in last:
        for value in (row["first"] if zone=="first" else [row["second"]]):
            counts[value-1]+=1
    return [(c+alpha*baseline)/(len(last)+alpha) for c in counts]

def repeat_scores(history,zone):
    n=38 if zone=="first" else 8
    recent=history[-5:]
    counts=[0.0]*n
    for a,b in zip(recent[:-1],recent[1:]):
        old=set(a["first"] if zone=="first" else [a["second"]])
        new=set(b["first"] if zone=="first" else [b["second"]])
        for num in old & new: counts[num-1]+=1
    # smoothed to avoid zero division; informational heuristic only
    return [(x+0.5)/5.0 for x in counts]

def fusion_scores(history,zone,weights=(0.5,0.35,0.15)):
    if len(history)<10:raise ValueError("Need 10 prior draws")
    if len(weights)!=3 or any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-9:
        raise ValueError("Weights must be three nonnegative values summing to 1")
    components=[score(history,zone),bayesian_scores(history,zone),repeat_scores(history,zone)]
    return [sum(weights[k]*components[k][i] for k in range(3)) for i in range(len(components[0]))]

def predict_fusion(history):
    first=fusion_scores(history,"first")
    second=fusion_scores(history,"second")
    result=pick(first,second)
    result.update({"through_draw":history[-1]["draw_id"],"feature_windows":[5,10],
                   "ensemble_weights":{"frequency":0.5,"bayesian":0.35,"repeat":0.15},
                   "ensemble_type":"deterministic_statistical_proxy",
                   "trained_ai_integrated":False,"real_qpu":False,
                   "real_historical_backtest_verified":False,"predictive_advantage_verified":False})
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    a=p.parse_args()
    print(json.dumps(predict_fusion(load_history(a.csv)),indent=2))
if __name__=="__main__":main()
