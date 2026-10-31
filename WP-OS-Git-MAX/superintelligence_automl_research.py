#!/usr/bin/env python3
"""Isolated TensorFlow/Keras-Tuner research bridge for WP-OS data.

This does not replace or feed the production WP-OS selection models.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random
from pathlib import Path

UPSTREAM_COMMIT = "da09a58ae32aed50cdf13caa8a5e2f4c795a324d"
SEED = 20261031

def load_draws(path):
    rows=[]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            first=[int(x) for x in row["first"].replace(",", " ").split()]
            second=int(row["second"])
            if len(first)!=6 or len(set(first))!=6 or not all(1<=n<=38 for n in first):
                raise ValueError("Invalid first zone")
            if not 1<=second<=8:
                raise ValueError("Invalid second zone")
            rows.append({"draw_id":str(row["draw_id"]),"first":first,"second":second})
    if len(rows)<12:
        raise ValueError("At least 12 completed draws required")
    return rows

def features(history):
    values=[]
    for window in (5,10):
        recent=history[-window:]
        first=[0.0]*38
        second=[0.0]*8
        for draw in recent:
            for n in draw["first"]: first[n-1]+=1/window
            second[draw["second"]-1]+=1/window
        values.extend(first+second)
    return values

def dataset(draws):
    x=[]; yf=[]; ys=[]
    for end in range(10,len(draws)):
        x.append(features(draws[:end]))
        target=draws[end]
        first=[0.0]*38
        for n in target["first"]: first[n-1]=1.0
        yf.append(first); ys.append(target["second"]-1)
    return x,yf,ys

def run(draws, output_dir):
    import numpy as np
    import tensorflow as tf
    import keras_tuner as kt
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    x,yf,ys=dataset(draws)
    x=np.asarray(x,dtype="float32")
    yf=np.asarray(yf,dtype="float32")
    ys=np.asarray(ys,dtype="int32")
    def build(hp):
        inputs=tf.keras.Input(shape=(92,),name="frequency_5_10")
        hidden=tf.keras.layers.Dense(hp.Choice("units",[16,32]),activation="relu")(inputs)
        first=tf.keras.layers.Dense(38,activation="sigmoid",name="first")(hidden)
        second=tf.keras.layers.Dense(8,activation="softmax",name="second")(hidden)
        model=tf.keras.Model(inputs,[first,second])
        model.compile(optimizer="adam",
            loss={"first":"binary_crossentropy","second":"sparse_categorical_crossentropy"})
        return model
    tuner=kt.RandomSearch(build,objective="val_loss",max_trials=1,seed=SEED,
                          directory=str(output_dir),project_name="wp_os_isolated",overwrite=True)
    tuner.search(x,[yf,ys],epochs=1,validation_split=0.2,verbose=0)
    model=tuner.get_best_models(1)[0]
    first,second=model.predict(np.asarray([features(draws)],dtype="float32"),verbose=0)
    first_scores={str(i+1):float(v) for i,v in enumerate(first[0])}
    second_scores={str(i+1):float(v) for i,v in enumerate(second[0])}
    ranked_first=sorted(range(1,39),key=lambda n:(-first_scores[str(n)],n))[:6]
    ranked_second=max(range(1,9),key=lambda n:(second_scores[str(n)],-n))
    payload={
      "schema_version":1,"mode":"isolated_tensorflow_keras_tuner_research",
      "upstream_commit":UPSTREAM_COMMIT,"through_draw":draws[-1]["draw_id"],
      "training_samples":len(x),"feature_windows":[5,10],"seed":SEED,
      "first":sorted(ranked_first),"second":ranked_second,
      "scores":{"first":first_scores,"second":second_scores},
      "tensorflow_automl_executed":True,"prediction_integration":False,
      "production_model_changed":False,"predictive_advantage_verified":False,
      "note":"One-trial runtime proof only; not an accuracy or lottery-advantage validation."
    }
    payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True); p.add_argument("--output",required=True)
    p.add_argument("--workdir",default="/tmp/wp-os-superintelligence-automl")
    args=p.parse_args()
    result=run(load_draws(args.csv),Path(args.workdir))
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__": main()
