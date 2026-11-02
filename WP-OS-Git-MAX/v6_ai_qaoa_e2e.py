#!/usr/bin/env python3
"""V6 local QAOA simulator with optional OpenFE feature modes.

No QPU or ABE. --openfe preserves the 30% fusion mode.
--openfe-three-views uses OpenFE/LightGBM scores from latest-10, latest-20 and
full-archive views at 100%, before the local statevector combination search.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path

def validate(path):
    rows=[];seen=set()
    with open(path,encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=["draw_id","first","second"]:raise ValueError("bad columns")
        for row in reader:
            k=row["draw_id"];a=[int(x) for x in row["first"].split()];b=int(row["second"])
            if (len(k)!=9 or not k.isdigit() or k in seen or len(a)!=6
                    or len(set(a))!=6 or any(n<1 or n>38 for n in a) or b<1 or b>8):
                raise ValueError("bad draw "+k)
            seen.add(k);rows.append((k,a,b))
    if len(rows)<10:raise ValueError("at least 10 draws required")
    for i in range(len(rows)-1):
        if int(rows[i][0])>=int(rows[i+1][0]):raise ValueError("draw order")
    return rows

def run(path,use_openfe=False,openfe_weight=.30,openfe_three_views=False):
    if not 0<=openfe_weight<=1:raise ValueError("openfe_weight must be between 0 and 1")
    rows=validate(path);five=rows[-5:];ten=rows[-10:];twenty=rows[-20:]
    scores={}
    for n in range(1,39):
        a=sum(n in x[1] for x in five);b=sum(n in x[1] for x in ten)
        scores[n]=.6*(a+1)/(5+38/6)+.4*(b+1)/(10+38/6)
    feature_layer={"enabled":False,"weight":0.0}
    if openfe_three_views:
        from v6_openfe_bridge import score_three_history_views
        scores,metadata=score_three_history_views(rows,n_jobs=1)
        feature_layer={"enabled":True,"weight":1.0,**metadata}
    elif use_openfe:
        from v6_openfe_bridge import score_first_zone
        learned,metadata=score_first_zone(rows,n_jobs=1)
        low=min(learned.values());high=max(learned.values());span=high-low or 1.0
        learned={n:(learned[n]-low)/span for n in learned}
        base_low=min(scores.values());base_high=max(scores.values());base_span=base_high-base_low or 1.0
        base={n:(scores[n]-base_low)/base_span for n in scores}
        scores={n:(1-openfe_weight)*base[n]+openfe_weight*learned[n] for n in scores}
        feature_layer={"enabled":True,"weight":openfe_weight,**metadata}
    pool=sorted(scores,key=lambda n:(-scores[n],n))[:8]
    costs=[]
    for mask in range(256):
        selected=[pool[i] for i in range(8) if mask>>i&1]
        costs.append(sum(scores[n] for n in selected) if len(selected)==6 else -10)
    best=None
    for j in range(1,17):
        gamma=j*math.pi/32;beta=j*math.pi/64
        state=[complex(math.cos(-gamma*c),math.sin(-gamma*c))/16 for c in costs]
        for bit in range(8):
            c=math.cos(beta);t=-1j*math.sin(beta)
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
               "latest_20":{"rows":len(twenty),"from":twenty[0][0],"to":twenty[-1][0]},
               "complete_archive":{"rows":len(rows),"from":rows[0][0],"to":rows[-1][0]},
               "full_archive":{"rows":len(rows),"from":rows[0][0],"to":rows[-1][0]}},
      "sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "model":"optional OpenFE/LightGBM three-history-view scoring + 8-qubit p=1 QAOA-inspired statevector; NOT QPU",
      "openfe_feature_layer":feature_layer,
      "status":"Research simulation; no evidence of predictive advantage"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--output")
    p.add_argument("--openfe",action="store_true");p.add_argument("--openfe-weight",type=float,default=.30)
    p.add_argument("--openfe-three-views",action="store_true")
    a=p.parse_args()
    result=run(a.csv,a.openfe,a.openfe_weight,a.openfe_three_views)
    out=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.output:Path(a.output).write_text(out,encoding="utf-8")
    print(out)
if __name__=="__main__":main()
