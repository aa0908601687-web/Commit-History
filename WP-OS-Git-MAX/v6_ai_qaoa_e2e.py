#!/usr/bin/env python3
"""Independent V6 AI/QAOA local simulator validation. No QPU, ABE or trained ML."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def validate(path):
    rows=[]; seen=set()
    with open(path,encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=["draw_id","first","second"]: raise ValueError("bad columns")
        for row in reader:
            k=row["draw_id"]; a=[int(x) for x in row["first"].split()]; b=int(row["second"])
            if len(k)!=9 or not k.isdigit() or k in seen or len(a)!=6 or len(set(a))!=6 or any(n<1 or n>38 for n in a) or b<1 or b>8: raise ValueError("bad draw "+k)
            seen.add(k);rows.append((k,a,b))
    if len(rows)<10:raise ValueError("at least 10 draws required")
    if any(int(rows[i][0])>=int(rows[i+1][0]) for i in range(len(rows)-1)):raise ValueError("draw order")
    return rows

def run(path):
    rows=validate(path); five=rows[-5:];ten=rows[-10:]
    scores={}
    for n in range(1,39):
        a=sum(n in x[1] for x in five);b=sum(n in x[1] for x in ten)
        scores[n]=.6*(a+1)/(5+38/6)+.4*(b+1)/(10+38/6)
    pool=sorted(scores,key=lambda n:(-scores[n],n))[:8]
    costs=[]
    for mask in range(256):
        selected=[pool[i] for i in range(8) if mask>>i&1]
        costs.append(sum(scores[n] for n in selected) if len(selected)==6 else -10)
    best=None
    for j in range(1,17):
        gamma=j*math.pi/32; beta=j*math.pi/64
        state=[complex(math.cos(-gamma*c),math.sin(-gamma*c))/16 for c in costs]
        for bit in range(8):
            c=math.cos(beta); t=-1j*math.sin(beta)
            for z in range(256):
                if not z>>bit&1:
                    other=z|(1<<bit);x,y=state[z],state[other]
                    state[z]=c*x+t*y;state[other]=t*x+c*y
        mask=max((z for z in range(256) if z.bit_count()==6),key=lambda z:(abs(state[z])**2,costs[z]))
        selected=sorted(pool[i] for i in range(8) if mask>>i&1)
        cand=(sum(scores[n] for n in selected),tuple(selected))
        if best is None or cand[0]>best[0]:best=cand
    second=max(range(1,9),key=lambda n:(.6*sum(x[2]==n for x in five)+.4*sum(x[2]==n for x in ten),-n))
    return {"first_zone":list(best[1]),"second_zone":second,"last_draw":rows[-1][0],
      "views":{"latest_5":{"rows":5,"from":five[0][0],"to":five[-1][0]},
               "latest_10":{"rows":10,"from":ten[0][0],"to":ten[-1][0]},
               "complete_archive":{"rows":len(rows),"from":rows[0][0],"to":rows[-1][0]}},
      "sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "model":"Bayesian-style heuristic and 8-qubit p=1 QAOA-inspired statevector; NOT trained ML/QPU",
      "feature_windows":"5 and 10 draws only; full archive for validation",
      "status":"Research simulation; no evidence of predictive advantage"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--output");a=p.parse_args()
    result=run(a.csv);out=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.output:Path(a.output).write_text(out,encoding="utf-8")
    print(out)
if __name__=="__main__":main()
