#!/usr/bin/env python3
"""Independent quantum-inspired lottery heuristic: no ABE or quantum hardware."""
import argparse, csv, json, random, math, hashlib
from collections import Counter

def load(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows=list(csv.DictReader(fh))
    draws=[]; ids=set()
    for row in rows:
        ident=row["draw_id"]
        first=tuple(map(int,row["first"].replace(",", " ").split()))
        second=int(row["second"])
        if ident in ids or len(first)!=6 or len(set(first))!=6 or any(not 1<=x<=38 for x in first) or not 1<=second<=8:
            raise ValueError("Invalid draw "+ident)
        ids.add(ident); draws.append((ident,first,second))
    if len(draws)<20: raise ValueError("At least 20 draws required")
    return draws

def select(draws,seed=20261031):
    if len(draws)<20: raise ValueError("At least 20 draws required")
    recent=draws[-20:]; freq=Counter(); pairs=Counter(); second=Counter()
    for i,(_,numbers,s) in enumerate(recent):
        w=.55+.45*(i+1)/20
        second[s]+=w
        for n in numbers: freq[n]+=w
        for j,a in enumerate(numbers):
            for b in numbers[j+1:]: pairs[tuple(sorted((a,b)))]+=w
    last=set(recent[-1][1]); rng=random.Random(seed)
    def objective(a):
        score=sum(freq[n] for n in a)+.085*sum(pairs[tuple(sorted((x,y)))] for j,x in enumerate(a) for y in a[j+1:])
        return score-3*max(0,len(last.intersection(a))-2)
    best=None; best_score=-float("inf")
    for _ in range(25):
        a=sorted(rng.sample(range(1,39),6)); current=objective(a)
        for i in range(1500):
            b=a.copy(); j=rng.randrange(6); n=rng.randrange(1,39)
            if n in b: continue
            b[j]=n; b.sort(); score=objective(b); temp=1.5*(.007**(i/1500))
            if score>current or rng.random()<math.exp((score-current)/temp): a,current=b,score
            if current>best_score: best,best_score=a.copy(),current
    zone2=max(range(1,9),key=lambda n:(second[n],-n))
    result={"system":"WP-AI Quantum Super Intelligence MAX","method":"simulated_annealing_heuristic","abe_used":False,"neural_models_executed":False,"quantum_hardware_used":False,"through_draw":recent[-1][0],"first":best,"second":zone2,"seed":seed,"feature_window":20,"objective_score":round(best_score,6)}
    result["sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--seed",type=int,default=20261031)
    args=p.parse_args()
    print(json.dumps(select(load(args.csv),args.seed),ensure_ascii=False,indent=2))
