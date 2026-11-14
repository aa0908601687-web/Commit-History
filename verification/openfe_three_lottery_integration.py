"""Independent synthetic-data integration test for three lottery configurations.
Not a prediction accuracy or live official data verification.
"""
import json
import random
from itertools import combinations
import numpy as np
import pandas as pd
from openfe import OpenFE, transform
from lightgbm import LGBMClassifier
import dimod
import cudaq

CONFIGS = {"power": (38, 6, 8), "lotto": (49, 6, None), "539": (39, 5, None)}

def features(history, n, max_number):
    last5, last10 = history[-5:], history[-10:]
    return {"number": float(n), "scaled": n/max_number,
            "freq5": float(sum(n in d for d in last5)),
            "freq10": float(sum(n in d for d in last10)),
            "odd": float(n % 2),
            "gap": float(next((i for i,d in enumerate(reversed(last10),1) if n in d),11))}

def run(kind):
    maximum, count, second_max = CONFIGS[kind]
    rng = random.Random(20261114 + maximum)
    draws = [tuple(sorted(rng.sample(range(1, maximum+1), count))) for _ in range(18)]
    x, y = [], []
    for end in range(10, len(draws)):
        for number in range(1, maximum+1):
            x.append(features(draws[:end], number, maximum))
            y.append(int(number in draws[end]))
    train = pd.DataFrame(x)
    labels = pd.Series(y)
    current = pd.DataFrame([features(draws, n, maximum) for n in range(1,maximum+1)])
    generated = OpenFE().fit(data=train,label=labels,task="classification",n_jobs=1,verbose=False)
    chosen = generated[:3]
    if chosen:
        train, current = transform(train,current,chosen,n_jobs=1)
    model = LGBMClassifier(n_estimators=25,num_leaves=7,max_depth=3,random_state=20261114,n_jobs=1,verbosity=-1)
    model.fit(train, labels)
    scores = model.predict_proba(current)[:,1]
    assert len(scores)==maximum and np.isfinite(scores).all()
    pool = sorted(range(1,maximum+1),key=lambda n:(-scores[n-1],n))[:8]
    # Exact enumerated optimum under linear QUBO objective and fixed cardinality.
    candidates = list(combinations(pool,count))
    exact = max(candidates,key=lambda c:(sum(float(scores[n-1]) for n in c),tuple(-n for n in c)))
    # QUBO solver on the same 8-bit candidate pool; cardinality penalty.
    penalty=10.0
    Q={}
    for i,n in enumerate(pool):
        Q[(i,i)] = -float(scores[n-1]) + penalty*(1-2*count)
        for j in range(i+1,len(pool)):
            Q[(i,j)] = 2*penalty
    best = dimod.ExactSolver().sample_qubo(Q).first.sample
    qubo_choice=tuple(sorted(pool[i] for i in range(8) if best[i]))
    assert len(qubo_choice)==count
    assert abs(sum(scores[n-1] for n in qubo_choice)-sum(scores[n-1] for n in exact))<1e-8
    cudaq.set_target("qpp-cpu")
    kernel=cudaq.make_kernel()
    q=kernel.qalloc()
    kernel.h(q)
    counts=cudaq.sample(kernel,shots_count=32)
    assert sum(counts.values())==32
    output={"game":kind,"first_zone":sorted(exact),"second_zone":1 if second_max else None,
            "history_source":"deterministic synthetic fixture","history_draws":len(draws),
            "openfe_generated":len(generated),"openfe_selected":len(chosen),
            "lightgbm_trained":True,"qubo_exact_agreement":True,
            "cudaq_cpu_executed":True,"cudaq_used_for_selection":False,
            "exact_candidates":len(candidates),"official_data_verified":False}
    assert len(output["first_zone"])==count
    assert all(1<=n<=maximum for n in output["first_zone"])
    if second_max: assert 1<=output["second_zone"]<=second_max
    return output

if __name__=="__main__":
    print(json.dumps([run(name) for name in CONFIGS],ensure_ascii=False))
