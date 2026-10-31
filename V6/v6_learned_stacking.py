"""Leakage-controlled chronological learned stacking for V6.
Three trained small nets, convex meta-weights learned on earlier rolling predictions.
Research-only, no real QPU, no ABE, no claimed predictive benefit.
"""
import argparse
import json
import math
from v6_e2e import load_history
from v6_ensemble_v2 import neural_model, normalize
from v6_lottery_pipeline import pick

SEEDS=(11,23,37)

def target(row):
    out=[0.0]*46
    for n in row["first"]:out[n-1]=1.0
    out[38+row["second"]-1]=1.0
    return out

def crossfit_predictions(draws, max_folds=8):
    if len(draws)<15:
        raise ValueError("At least 15 complete draws are required")
    start=max(12,len(draws)-max_folds)
    folds=[]
    for t in range(start,len(draws)):
        # Only rows before t are used to train the base models.
        member=[neural_model(draws[:t],seed) for seed in SEEDS]
        folds.append((member,target(draws[t]),draws[t]["draw_id"]))
    return folds

def learn_weights(folds,steps=80,rate=0.4):
    # Fully chronological meta-training; no test draw used at its prediction step.
    logits=[0.0]*len(SEEDS)
    for _ in range(steps):
        z=[math.exp(a-max(logits)) for a in logits]
        weights=[a/sum(z) for a in z]
        grad=[0.0]*len(SEEDS)
        for models,y,_ in folds:
            for i in range(46):
                p=min(max(sum(weights[m]*models[m][i] for m in range(len(SEEDS))),1e-6),1-1e-6)
                deriv=(p-y[i])/(p*(1-p))
                for m in range(len(SEEDS)):
                    grad[m]+=deriv*weights[m]*(models[m][i]-p)
        scale=len(folds)*46
        logits=[min(10,max(-10,logits[m]-rate*grad[m]/scale)) for m in range(len(SEEDS))]
    z=[math.exp(a-max(logits)) for a in logits]
    return [v/sum(z) for v in z]

def predict(draws):
    folds=crossfit_predictions(draws)
    weights=learn_weights(folds)
    members=[neural_model(draws,seed) for seed in SEEDS]
    merged=[sum(weights[k]*members[k][i] for k in range(3)) for i in range(46)]
    selection=pick(normalize(merged[:38]),normalize(merged[38:]))
    selection.update({"through_draw":draws[-1]["draw_id"],
        "architecture":"three_dense_nets_plus_chronological_learned_stacking",
        "learned_stacking_weights":dict(zip(map(str,SEEDS),weights)),
        "meta_training_folds":len(folds),"meta_last_target_draw":folds[-1][2],
        "meta_training_only_on_past_draws":True,
        "feature_windows":[5,10],"trained_ai_integrated":True,
        "real_historical_validation_performed":False,"predictive_advantage_verified":False,
        "real_qpu":False,"abe_integrated":False})
    return selection

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    args=p.parse_args()
    print(json.dumps(predict(load_history(args.csv)),indent=2))
if __name__=="__main__":main()
