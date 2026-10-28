#!/usr/bin/env python3
"""Experimental fourth model: conservative adaptive weighting of 5/10 and 10/20 scores.

No claim of lottery predictability. Uses only information preceding each target draw.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from collections import Counter

BASE = Path(__file__).with_name("wp_os_git_max.py")
spec = importlib.util.spec_from_file_location("wp_os_git_max_adaptive_base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def rank_zone(draws, upper, short, long, scalar=False):
    if len(draws) < long:
        raise ValueError("Need at least %d completed draws" % long)
    recent = draws[-long:]
    recent_short = draws[-short:]
    if scalar:
        cshort, clong = Counter(recent_short), Counter(recent)
        last = {recent[-1]}
    else:
        cshort = Counter(n for row in recent_short for n in row)
        clong = Counter(n for row in recent for n in row)
        last = set(recent[-1])
    scores = {}
    for n in range(1, upper + 1):
        age = next((i for i, row in enumerate(reversed(recent))
                    if (row == n if scalar else n in row)), long)
        scores[n] = (.45*cshort[n]/short + .25*clong[n]/long
                     + .15*(clong[n]+1)/(long+upper)
                     + .10*(n in last) + .05*min(age,long)/long)
    return scores


def normalized(scores):
    # Relative heuristics; NOT calibrated probabilities.
    total = sum(max(0.0, x) for x in scores.values())
    if total == 0:
        return {k: 1/len(scores) for k in scores}
    return {k: max(0.0, v)/total for k, v in scores.items()}


def losses_for_previous_draws(draws, scalar, upper):
    """Prequential Brier losses, never using the target draw for fitting its weights."""
    a, b = [], []
    for end in range(20, len(draws)):
        history, observed = draws[:end], draws[end]
        s = [normalized(rank_zone(history, upper, w1, w2, scalar))
             for w1, w2 in ((5,10),(10,20))]
        actual = {observed} if scalar else set(observed)
        for losses, p in zip((a,b),s):
            losses.append(sum((p[n] - (1/len(actual) if n in actual else 0))**2
                              for n in p))
    return a,b


def weight(draws, scalar, upper):
    a,b = losses_for_previous_draws(draws, scalar, upper)
    if len(a) < 10:
        return .5
    # Bayesian-style shrinkage around equal prior odds, not a full probabilistic BMA.
    diff = sum(y-x for x,y in zip(a,b))
    log_odds = max(-1.1, min(1.1, 8.0*diff))
    raw = 1/(1+math.exp(-log_odds))
    strength = len(a)/(len(a)+30)
    return round(.5 + strength*(raw-.5), 8)


def predict(draws):
    if len(draws) < 20:
        raise ValueError("At least 20 completed draws required")
    results = {}
    weights = {}
    for key, upper, scalar in (("first",38,False),("second",8,True)):
        seq = [d[key] for d in draws]
        w = weight(seq,scalar,upper)
        a = normalized(rank_zone(seq,upper,5,10,scalar))
        b = normalized(rank_zone(seq,upper,10,20,scalar))
        fused = {n:w*a[n]+(1-w)*b[n] for n in a}
        chosen = sorted(fused,key=lambda n:(-fused[n],n))[:(1 if scalar else 6)]
        results[key] = chosen[0] if scalar else sorted(chosen)
        weights[key] = {"5_10":w,"10_20":round(1-w,8),"training_targets":max(0,len(seq)-20)}
    payload = {"model":"Adaptive Bayesian Ensemble MAX (experimental)",
               "through_draw":draws[-1]["draw_id"], **results, "weights":weights,
               "method":"Conservative prequential Brier-loss weighting; not validated prediction"}
    payload["sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload


def backtest(draws):
    rows = []
    for end in range(20,len(draws)):
        pred = predict(draws[:end])
        target = draws[end]
        hits = len(set(pred["first"]) & set(target["first"]))
        second_hit = pred["second"] == target["second"]
        rows.append({"target_draw":target["draw_id"],"through_draw":pred["through_draw"],
                     "first_zone_hits":hits,"second_zone_hit":bool(second_hit),
                     "jackpot_hit":bool(hits == 6 and second_hit),
                     "weights":pred["weights"]})
    count = Counter(x["first_zone_hits"] for x in rows)
    return {"model":"Adaptive Bayesian Ensemble MAX (experimental)","tested":len(rows),
            "first_zone_hits":{str(i):count[i] for i in range(7)},
            "first_zone_3plus_rate":sum(x["first_zone_hits"]>=3 for x in rows)/len(rows) if rows else None,
            "second_zone_hits":sum(x["second_zone_hit"] for x in rows),
            "jackpot_hits":sum(x["jackpot_hit"] for x in rows),
            "fair_jackpot_probability":1/(math.comb(38,6)*8),"per_draw":rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--backtest",action="store_true")
    args = p.parse_args()
    draws = base.load(args.csv)
    print(json.dumps(backtest(draws) if args.backtest else predict(draws),
                     ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
