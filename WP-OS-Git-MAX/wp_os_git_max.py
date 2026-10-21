#!/usr/bin/env python3
"""WP-OS Git MAX: small, reproducible 5/10-draw research baseline."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter

def load(path):
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            first = tuple(sorted(int(x) for x in row["first"].replace(",", " ").split()))
            second = int(row["second"])
            if len(first) != 6 or len(set(first)) != 6 or not all(1 <= x <= 38 for x in first) or not 1 <= second <= 8:
                raise ValueError("Invalid draw: " + str(row.get("draw_id")))
            out.append({"draw_id":row["draw_id"], "first":first, "second":second})
    if len(set(d["draw_id"] for d in out)) != len(out):
        raise ValueError("Duplicate draw IDs")
    return out

def score(seq, upper):
    if len(seq) < 10:
        raise ValueError("Need 10 prior draws")
    last5, last10 = seq[-5:], seq[-10:]
    counts5, counts10 = Counter(last5), Counter(last10)
    ranked = []
    for n in range(1, upper + 1):
        p5 = counts5[n]/5
        p10 = counts10[n]/10
        prior = (counts10[n] + 1)/(10 + upper)
        repeat = float(n in last10[-1])
        ago = next((i for i, x in enumerate(reversed(last10)) if x == n), 10)
        recency = min(ago, 10)/10
        s = .45*p5 + .25*p10 + .15*prior + .10*repeat + .05*recency
        ranked.append((n, round(s, 12)))
    return sorted(ranked, key=lambda x:(-x[1], x[0]))

def predict(draws):
    if len(draws) < 10:
        raise ValueError("At least 10 completed draws are required")
    zone1 = [n for d in draws for n in d["first"]]
    # Frequency is measured per DRAW, not per flattened number position.
    f = [d["first"] for d in draws]
    l = [d["second"] for d in draws]
    ranking1 = rank_draws(f, 38)
    ranking2 = score(l, 8)
    first = sorted(n for n, _ in ranking1[:6])
    pairs = Counter(pair for d in draws[-10:] for pair in itertools.combinations(d["first"], 2))
    payload = {"through_draw":draws[-1]["draw_id"],"first":first,"second":ranking2[0][0],
        "top_first_scores":ranking1[:10], "second_scores":ranking2,
        "top_pairs":[{"pair":list(p),"count":c} for p,c in sorted(pairs.items(), key=lambda z:(-z[1],z[0]))[:10]],
        "method":"Fixed 5/10-draw research baseline, no predictive claim"}
    payload["sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return payload

def rank_draws(draws, upper):
    if len(draws) < 10: raise ValueError("Need 10 completed draws")
    a,b = draws[-5:],draws[-10:]
    c5=Counter(n for d in a for n in d)
    c10=Counter(n for d in b for n in d)
    last=set(b[-1])
    vals=[]
    for n in range(1,upper+1):
        age=next((i for i,draw in enumerate(reversed(b)) if n in draw),10)
        s=.45*c5[n]/5 + .25*c10[n]/10 + .15*(c10[n]+1)/(10+upper) + .10*(n in last)+.05*min(age,10)/10
        vals.append((n,round(s,12)))
    return sorted(vals,key=lambda x:(-x[1],x[0]))

def backtest(draws):
    counts=Counter()
    hit_second=0
    for end in range(10,len(draws)):
        pred=predict(draws[:end])
        actual=draws[end]
        hits=len(set(pred["first"]) & set(actual["first"]))
        counts[hits]+=1
        hit_second += pred["second"] == actual["second"]
    return {"tested":max(0,len(draws)-10), "first_zone_hits":{str(n):counts[n] for n in range(7)},
      "second_zone_hits":hit_second, "random_first_zone_3plus":sum(math.comb(6,k)*math.comb(32,6-k) for k in range(3,7))/math.comb(38,6),
      "fair_jackpot_probability":1/(math.comb(38,6)*8)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--backtest",action="store_true")
    args=p.parse_args()
    draws=load(args.csv)
    print(json.dumps(backtest(draws) if args.backtest else predict(draws),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
