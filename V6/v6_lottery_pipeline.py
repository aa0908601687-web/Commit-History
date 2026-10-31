"""V6 two-zone lottery selection with small QAOA shortlisting.

Research-only: 38 choose 6 + 8 choose 1. Input scores are supplied, not predicted.
No ABE, real QPU, or assertion of predictive advantage.
"""
import argparse
import json
import math
from v6_local_qaoa import solve

def pick(first_scores, second_scores, shortlist=8):
    if len(first_scores)!=38 or len(second_scores)!=8:
        raise ValueError("Need 38 first-zone scores and 8 second-zone scores")
    values=[float(v) for v in first_scores+second_scores]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Scores must be finite")
    if not 6<=shortlist<=10:
        raise ValueError("QAOA local shortlist must be between 6 and 10")
    order=sorted(range(38),key=lambda i:(-first_scores[i],i))[:shortlist]
    local=solve([first_scores[i] for i in order],choose=6)
    selected=sorted(order[i]+1 for i in local["selected_indices"])
    special=max(range(8),key=lambda i:(second_scores[i],-i))+1
    return {"first":selected,"second":special,"shortlist_size":shortlist,
            "method":"qaoa_shortlist_plus_second_zone_argmax",
            "full_38_qubit_qaoa":False,"real_qpu":False,
            "scores_supplied_externally":True,
            "predictive_advantage_verified":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first-scores",required=True,help="38 comma-separated values")
    p.add_argument("--second-scores",required=True,help="8 comma-separated values")
    a=p.parse_args()
    parse=lambda s:[float(v) for v in s.split(",")]
    print(json.dumps(pick(parse(a.first_scores),parse(a.second_scores)),indent=2))

if __name__=="__main__":
    main()
