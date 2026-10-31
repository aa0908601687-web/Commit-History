"""V6 experimental adaptive ensemble v2 (no repeat-number feature).

Weights are research score weights, NOT calibrated lottery probabilities.
Uses only trailing 5/10 draws and deterministic trainable neural networks.
Bayesian and uncertainty terms are separate heuristic score components.
"""
import argparse
import json
import math
import random
from v6_e2e import load_history, score
from v6_ai_fusion import bayesian_scores
from v6_lottery_pipeline import pick
from v6_neural_train import feature, dataset, sigmoid

WEIGHTS={"bayesian":0.25,"deep_ensemble":0.25,"stacking":0.20,"frequency":0.15,"stability":0.15}
assert abs(sum(WEIGHTS.values())-1)<1e-12

def neural_model(draws, seed, hidden=8, epochs=3, lr=0.025):
    xs,ys=dataset(draws)
    if len(xs)<2:raise ValueError("Need at least 12 completed draws")
    rng=random.Random(seed)
    w1=[[rng.uniform(-0.03,0.03) for _ in range(92)] for _ in range(hidden)]
    b1=[0.0]*hidden
    w2=[[rng.uniform(-0.03,0.03) for _ in range(hidden)] for _ in range(46)]
    b2=[0.0]*46
    for _ in range(epochs):
        for x,y in zip(xs,ys):
            h=[math.tanh(sum(a*v for a,v in zip(row,x))+bias) for row,bias in zip(w1,b1)]
            output=[sigmoid(sum(a*v for a,v in zip(row,h))+bias) for row,bias in zip(w2,b2)]
            err=[output[k]-y[k] for k in range(46)]
            grad=[sum(err[k]*w2[k][j] for k in range(46))*(1-h[j]**2) for j in range(hidden)]
            for k in range(46):
                for j in range(hidden):w2[k][j]-=lr*err[k]*h[j]
                b2[k]-=lr*err[k]
            for j in range(hidden):
                for t in range(92):w1[j][t]-=lr*grad[j]*x[t]
                b1[j]-=lr*grad[j]
    x=feature(draws)
    h=[math.tanh(sum(a*v for a,v in zip(row,x))+bias) for row,bias in zip(w1,b1)]
    return [sigmoid(sum(a*v for a,v in zip(row,h))+bias) for row,bias in zip(w2,b2)]

def normalize(values):
    lo,hi=min(values),max(values)
    if hi-lo<1e-12:return [0.5]*len(values)
    return [(x-lo)/(hi-lo) for x in values]

def predict(draws):
    if len(draws)<12:raise ValueError("At least 12 historical draws required")
    models=[neural_model(draws,seed) for seed in (11,23,37)]
    output={}
    for zone,sl in (("first",slice(0,38)),("second",slice(38,46))):
        members=[m[sl] for m in models]
        mean=[sum(x[i] for x in members)/len(members) for i in range(len(members[0]))]
        disagreement=[sum((x[i]-mean[i])**2 for x in members)/len(members) for i in range(len(mean))]
        # Stacking proxy is a fixed weighted blending of distinct trained nets;
        # NOT learned out-of-sample stacking weights.
        stack=[0.5*members[0][i]+0.3*members[1][i]+0.2*members[2][i] for i in range(len(mean))]
        freq=score(draws,zone)
        bayes=bayesian_scores(draws,zone)
        # High disagreement incurs penalty, not a claim of confidence calibration.
        stability=[1/(1+disagreement[i]) for i in range(len(mean))]
        components={"bayesian":bayes,"deep_ensemble":mean,"stacking":stack,
                    "frequency":freq,"stability":stability}
        normalized={name:normalize(vals) for name,vals in components.items()}
        output[zone]=[sum(WEIGHTS[name]*normalized[name][i] for name in WEIGHTS)
                      for i in range(len(mean))]
    chosen=pick(output["first"],output["second"])
    chosen.update({"through_draw":draws[-1]["draw_id"],"weights":WEIGHTS,
                   "mode":"research_trainable_ensemble_fixed_stack_proxy",
                   "trained_neural_models":3,"repetition_weight":0,
                   "stacking_weights_learned":False,
                   "probability_calibrated":False,
                   "real_data_validated":False,
                   "predictive_advantage_verified":False,
                   "real_qpu":False,"abe_integrated":False})
    return chosen

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",required=True)
    args=parser.parse_args()
    print(json.dumps(predict(load_history(args.csv)),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
