"""Train an actual small neural net on historical 5/10-draw features.
Python standard library only. Research only, NOT a demonstrated lottery predictor.
"""
import argparse,json,math,random
from v6_e2e import load_history
from v6_lottery_pipeline import pick

def feature(rows):
    out=[]
    for size in (5,10):
        part=rows[-size:]
        first=[0.0]*38; second=[0.0]*8
        for r in part:
            for v in r["first"]:first[v-1]+=1/size
            second[r["second"]-1]+=1/size
        out+=first+second
    return out

def dataset(draws):
    xs=[];ys=[]
    for i in range(10,len(draws)):
        xs.append(feature(draws[:i]))
        target=[0.0]*46
        for v in draws[i]["first"]:target[v-1]=1.0
        target[38+draws[i]["second"]-1]=1.0
        ys.append(target)
    return xs,ys

def sigmoid(x):
    if x>=0:return 1/(1+math.exp(-x))
    e=math.exp(x);return e/(1+e)

def train(draws,epochs=8,hidden=12,lr=0.025,seed=20261101):
    if len(draws)<12:raise ValueError("Require 12+ historical draws")
    if not (1<=epochs<=100 and 1<=hidden<=64 and 0<lr<=1):
        raise ValueError("Invalid hyperparameters")
    xs,ys=dataset(draws)
    rng=random.Random(seed)
    w1=[[rng.uniform(-0.03,0.03) for _ in range(92)] for _ in range(hidden)]
    b1=[0.0]*hidden
    w2=[[rng.uniform(-0.03,0.03) for _ in range(hidden)] for _ in range(46)]
    b2=[0.0]*46
    for _ in range(epochs):
        for x,y in zip(xs,ys):
            h=[math.tanh(sum(v*a for v,a in zip(row,x))+bias) for row,bias in zip(w1,b1)]
            out=[sigmoid(sum(v*a for v,a in zip(row,h))+bias) for row,bias in zip(w2,b2)]
            error=[out[k]-y[k] for k in range(46)]
            dh=[sum(error[k]*w2[k][j] for k in range(46))*(1-h[j]*h[j]) for j in range(hidden)]
            for k in range(46):
                for j in range(hidden):w2[k][j]-=lr*error[k]*h[j]
                b2[k]-=lr*error[k]
            for j in range(hidden):
                for t in range(92):w1[j][t]-=lr*dh[j]*x[t]
                b1[j]-=lr*dh[j]
    x=feature(draws)
    h=[math.tanh(sum(v*a for v,a in zip(row,x))+bias) for row,bias in zip(w1,b1)]
    scores=[sigmoid(sum(v*a for v,a in zip(row,h))+bias) for row,bias in zip(w2,b2)]
    selection=pick(scores[:38],scores[38:])
    return {"model":"trained_dense_neural_network_92_12_46",
            "training_samples":len(xs),"epochs":epochs,"seed":seed,
            "trained_ai_integrated":True,"first":selection["first"],"second":selection["second"],
            "through_draw":draws[-1]["draw_id"],"feature_windows":[5,10],
            "evaluation_on_unseen_real_draws":False,"predictive_advantage_verified":False,
            "real_qpu":False,"abe_integrated":False}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",required=True)
    args=parser.parse_args()
    print(json.dumps(train(load_history(args.csv)),indent=2))
if __name__=="__main__":main()
