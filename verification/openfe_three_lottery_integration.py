"""Independent synthetic-data integration test for three lottery configurations.
Not a prediction accuracy or live official data verification.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
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

def read_real_csv(csv_path, kind):
    """Validate chronology and legal lottery zones. Does not certify provenance."""
    maximum, count, second_max = CONFIGS[kind]
    rows, second_rows = [], []
    previous = -1
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"draw_id", "first"} | ({"second"} if second_max else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"required CSV columns: {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            try:
                draw_id = int(row["draw_id"])
                first = tuple(sorted(int(v) for v in row["first"].replace(",", " ").split()))
                second = int(row["second"]) if second_max else None
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid CSV on line {line}") from exc
            if draw_id <= previous:
                raise ValueError(f"non-increasing draw ID on line {line}")
            if len(first) != count or len(set(first)) != count or not all(1 <= n <= maximum for n in first):
                raise ValueError(f"invalid first-zone numbers on line {line}")
            if second_max and not (1 <= second <= second_max):
                raise ValueError(f"invalid second-zone number on line {line}")
            rows.append(first)
            if second_max:
                second_rows.append(second)
            previous = draw_id
    if len(rows) < 16:
        raise ValueError("at least 16 completed draws are needed to train all views")
    return rows, second_rows, str(previous)


def run(kind, csv_path=None):
    maximum, count, second_max = CONFIGS[kind]
    if csv_path:
        draws, second_history, cutoff = read_real_csv(csv_path, kind)
    else:
        rng = random.Random(20261114 + maximum)
        draws = [tuple(sorted(rng.sample(range(1, maximum+1), count))) for _ in range(18)]
        second_history, cutoff = [], None
    # Three independently trained historical perspectives. Every row uses only
    # completed draws preceding its label. The archive view uses all anchors;
    # no prediction feature looks back beyond ten draws.
    views = {"latest_5": 5, "latest_10": 10, "full_archive": None}
    weights = {"latest_5": 0.35, "latest_10": 0.35, "full_archive": 0.30}
    by_view, feature_counts, anchor_counts = {}, {}, {}
    for view, anchor_limit in views.items():
        ends = list(range(10, len(draws)))
        if anchor_limit is not None:
            ends = ends[-anchor_limit:]
        x, y = [], []
        for anchor in ends:
            for number in range(1, maximum+1):
                row = features(draws[:anchor], number, maximum)
                # Preserve distinct 5/10 feature sets for the short views.
                if view == "latest_5":
                    row.pop("freq10")
                    row.pop("gap")
                elif view == "latest_10":
                    row.pop("freq5")
                x.append(row)
                y.append(int(number in draws[anchor]))
        train = pd.DataFrame(x)
        labels = pd.Series(y)
        current_rows = []
        for number in range(1, maximum+1):
            row = features(draws, number, maximum)
            if view == "latest_5":
                row.pop("freq10")
                row.pop("gap")
            elif view == "latest_10":
                row.pop("freq5")
            current_rows.append(row)
        current = pd.DataFrame(current_rows)
        assert set(labels.unique()) == {0, 1}
        generated = OpenFE().fit(data=train,label=labels,task="classification",n_jobs=1,verbose=False)
        chosen = generated[:2]
        if chosen:
            train, current = transform(train,current,chosen,n_jobs=1)
        model = LGBMClassifier(n_estimators=25,num_leaves=7,max_depth=3,random_state=20261114,n_jobs=1,verbosity=-1)
        model.fit(train, labels)
        probabilities = model.predict_proba(current)[:,1]
        assert len(probabilities)==maximum and np.isfinite(probabilities).all()
        by_view[view] = probabilities
        feature_counts[view] = {"generated":len(generated),"selected":len(chosen)}
        anchor_counts[view] = len(ends)
    assert abs(sum(weights.values())-1.0)<1e-12
    scores = sum(weights[view]*by_view[view] for view in views)
    assert np.allclose(scores,0.35*by_view["latest_5"]+0.35*by_view["latest_10"]+0.30*by_view["full_archive"])
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
    second_pick = None
    if second_max:
        # This independent second-zone baseline is data-driven, not a trained OpenFE model.
        second_pick = max(range(1, second_max+1),
            key=lambda n: (0.6*sum(v == n for v in second_history[-5:]) +
                           0.4*sum(v == n for v in second_history[-10:]), -n)) if csv_path else 1
    output={"game":kind,"first_zone":sorted(exact),"second_zone":second_pick,
            "history_source":"supplied CSV, independent source verification pending" if csv_path else "deterministic synthetic fixture",
            "data_cutoff":cutoff,"data_sha256":hashlib.sha256(Path(csv_path).read_bytes()).hexdigest() if csv_path else None,
            "synthetic_data":not bool(csv_path),"history_draws":len(draws),
            "three_view_weights":weights,"view_anchor_counts":anchor_counts,
            "view_feature_counts":feature_counts,"three_view_fusion_executed":True,
            "lightgbm_trained":True,"qubo_exact_agreement":True,
            "cudaq_cpu_executed":True,"cudaq_used_for_selection":False,
            "exact_candidates":len(candidates),"official_data_verified":False}
    assert len(output["first_zone"])==count
    assert all(1<=n<=maximum for n in output["first_zone"])
    if second_max: assert 1<=output["second_zone"]<=second_max
    return output

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Independent three-lottery research; real-CSV inputs are not certified")
    parser.add_argument("--game", choices=tuple(CONFIGS), help="One lottery; required with --csv")
    parser.add_argument("--csv", help="Historical CSV to analyze in chronological draw order")
    args = parser.parse_args()
    if bool(args.game) != bool(args.csv):
        parser.error("--game and --csv must be supplied together")
    selected = [run(args.game, args.csv)] if args.game else [run(name) for name in CONFIGS]
    print(json.dumps(selected,ensure_ascii=False))
