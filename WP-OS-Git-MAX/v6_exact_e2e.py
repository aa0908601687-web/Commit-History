#!/usr/bin/env python3
"""V6 deterministic exact-search selector with optional OpenFE feature modes.

No QPU, ABE, Bayesian calibration layer, or V3 cache dependency. --openfe preserves the 30% fusion mode.
--openfe-three-views uses OpenFE/LightGBM scores from latest-5, latest-10 and
full-archive training views (all selection features capped at 10 draws) at 100%, before exhaustive eight-choose-six combination search.
"""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
# Support test runners that load this module directly by file path.
_module_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, _module_dir)

def validate(path):
    rows=[];seen=set()
    with open(path,encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ["draw_id", "first", "second"]:
            raise ValueError("bad columns")
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

def complete_result(path,rows,scores,feature_layer,high_fidelity_verify=False,use_cudaq=False,require_cudaq=False,ising_qubo_check=False,ortools_check=False):
    """Finish deterministic exact-search selection from supplied scores."""
    five=rows[-5:];ten=rows[-10:]
    if set(scores) != set(range(1, 39)) or any(not math.isfinite(scores[n]) for n in scores):
        raise ValueError("first-zone feature scores must be finite for every number 1..38")
    pool=sorted(scores,key=lambda n:(-scores[n],n))[:8]
    # Exactly enumerate the eight-choose-six space; no EXACT circuit or mixer.
    from v6_exact_search import exact_select
    exact_numbers, exact_score, count = exact_select(pool, scores)
    if count != 28:
        raise RuntimeError("unexpected exact-search candidate count")
    best = (exact_score, tuple(exact_numbers))
    cudaq_layer={"enabled":False,"executed":False,"qpu":False}
    if use_cudaq or require_cudaq:
        from v6_cudaq_backend import select_six
        cudaq_selected,cudaq_layer=select_six(pool,scores,required=require_cudaq)
        if cudaq_selected is not None:
            best=(sum(scores[n] for n in cudaq_selected),tuple(cudaq_selected))
    from v6_exact_search import compare_selection
    optimization_layer=compare_selection(pool,scores,best[1])
    best=(optimization_layer["exact_score"],tuple(optimization_layer["exact_selection"]))
    ising_layer={"enabled":False,"executed":False,"nvidia_native_solver_executed":False}
    if ising_qubo_check:
        # Independent CPU QUBO reference: fail closed on any objective/pool mismatch.
        root=Path(__file__).resolve().parents[1]
        verification_dir=str(root / "verification")
        if verification_dir not in sys.path:sys.path.insert(0, verification_dir)
        from v6_ising_qubo_reference import exact_six, check_solver_output
        ordered_scores=[float(scores[n]) for n in pool]
        qresult=exact_six(ordered_scores)
        chosen=[pool[i] for i in qresult["indices"]]
        bits=[int(i in qresult["indices"]) for i in range(8)]
        check=check_solver_output(bits,ordered_scores)
        if (qresult["combinations_checked"]!=28 or not check["optimal"]
                or not math.isclose(sum(scores[n] for n in chosen),
                                    optimization_layer["exact_score"],rel_tol=1e-10,abs_tol=1e-8)):
            raise RuntimeError("Ising/QUBO CPU reference disagrees with exact-search objective")
        ising_layer={"enabled":True,"executed":True,"backend":"python-cpu-qubo-reference",
                     "combinations_checked":28,"pool":list(pool),"reference_selection":chosen,
                     "reference_energy":qresult["energy"],
                     "same_optimal_score":True,
                     "same_selection":sorted(chosen)==sorted(optimization_layer["exact_selection"]),
                     "nvidia_native_solver_executed":False,
                     "rocq_runtime_proof":False,
                     "predictive_advantage_verified":False}
    ortools_layer={"enabled":False,"executed":False}
    if ortools_check:
        from v6_five_star_stack import ortools_six
        selected, value = ortools_six(pool, scores)
        if not math.isclose(value, optimization_layer["exact_score"], rel_tol=1e-10, abs_tol=1e-6):
            raise RuntimeError("OR-Tools CP-SAT disagrees with exact enumeration score")
        ortools_layer={"enabled":True,"executed":True,"backend":"ortools-cp-sat-cpu",
                       "selection":selected,"score":value,"same_optimal_score":True,
                       "same_selection":selected==list(optimization_layer["exact_selection"]),
                       "objective_scaled_to_integer":True}
    second=max(range(1,9),key=lambda n:(.6*sum(x[2]==n for x in five)+.4*sum(x[2]==n for x in ten),-n))
    fidelity_layer={"enabled":False,"scope":"legacy statevector verification removed"}
    return {"candidate_pool":pool,"first_zone":list(best[1]),"second_zone":second,"last_draw":rows[-1][0],
      "views":{"latest_5":{"rows":5,"from":five[0][0],"to":five[-1][0]},
               "latest_10":{"rows":10,"from":ten[0][0],"to":ten[-1][0]},
               "complete_archive":{"rows":len(rows),"from":rows[0][0],"to":rows[-1][0]},
               "full_archive":{"rows":len(rows),"from":rows[0][0],"to":rows[-1][0]}},
      "sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "model":"optional OpenFE/LightGBM three-history-view scoring + exact eight-choose-six selection; NOT QPU",
      "openfe_feature_layer":feature_layer,
      "high_fidelity_layer":fidelity_layer,
      "cudaq_layer":cudaq_layer,
      "ising_qubo_layer":ising_layer,
      "ortools_layer":ortools_layer,
      "exact_optimization_layer":optimization_layer,
      "status":"Research simulation; no evidence of predictive advantage"}

def run(path,use_openfe=False,openfe_weight=.30,openfe_three_views=False,high_fidelity_verify=False,use_cudaq=False,require_cudaq=False,ising_qubo_check=False,tabprep_three_views=False,ortools_check=False):
    if not 0<=openfe_weight<=1:raise ValueError("openfe_weight must be between 0 and 1")
    if tabprep_three_views and (openfe_three_views or use_openfe ):
        raise ValueError("--tabprep-three-views is an isolated experimental alternative to OpenFE scoring")
    rows=validate(path);five=rows[-5:];ten=rows[-10:]
    scores={}
    for n in range(1,39):
        a=sum(n in x[1] for x in five);b=sum(n in x[1] for x in ten)
        scores[n]=.6*(a+1)/(5+38/6)+.4*(b+1)/(10+38/6)
    feature_layer={"enabled":False,"weight":0.0}
    if tabprep_three_views:
        from v6_tabprep_bridge import score_three_history_views_tabprep
        scores, metadata = score_three_history_views_tabprep(rows)
        feature_layer = {"enabled": True, "weight": 1.0, **metadata}
    elif openfe_three_views:
        from v6_openfe_bridge import score_three_history_views
        scores,metadata=score_three_history_views(rows,n_jobs=1)
        feature_layer={"enabled":True,"weight":1.0,**metadata}
        feature_layer["fused_scores"] = dict(scores)
    elif use_openfe:
        from v6_openfe_bridge import score_first_zone
        learned,metadata=score_first_zone(rows,n_jobs=1)
        low=min(learned.values());high=max(learned.values());span=high-low or 1.0
        learned={n:(learned[n]-low)/span for n in learned}
        base_low=min(scores.values());base_high=max(scores.values());base_span=base_high-base_low or 1.0
        base={n:(scores[n]-base_low)/base_span for n in scores}
        scores={n:(1-openfe_weight)*base[n]+openfe_weight*learned[n] for n in scores}
        feature_layer={"enabled":True,"weight":openfe_weight,**metadata}
    return complete_result(path,rows,scores,feature_layer,high_fidelity_verify,use_cudaq,require_cudaq,ising_qubo_check,ortools_check)

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--output")
    p.add_argument("--openfe",action="store_true");p.add_argument("--openfe-weight",type=float,default=.30)
    p.add_argument("--openfe-three-views",action="store_true")
    p.add_argument("--tabprep-three-views",action="store_true",help="experimental TabPrep+LightGBM alternative; requires upstream TabPrep/AutoGluon installation")
    p.add_argument("--high-fidelity-verify",action="store_true",
                   help="deprecated no-op (EXACT statevector removed)")
    p.add_argument("--cudaq",action="store_true",
                   help="use optional NVIDIA CUDA-Q qpp-cpu local simulator; never a real QPU")
    p.add_argument("--require-cudaq",action="store_true",
                   help="fail instead of falling back when CUDA-Q is unavailable")
    p.add_argument("--ising-qubo-check",action="store_true",help="run fail-closed CPU QUBO reference against production exact-search score; not NVIDIA native solver")
    p.add_argument("--ortools-check",action="store_true",help="run OR-Tools CP-SAT CPU against exact reference; requires ortools")
    a=p.parse_args()
    result=run(a.csv,a.openfe,a.openfe_weight,a.openfe_three_views,a.high_fidelity_verify,a.cudaq or a.require_cudaq,a.require_cudaq,a.ising_qubo_check,a.tabprep_three_views,a.ortools_check)
    out=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.output:Path(a.output).write_text(out,encoding="utf-8")
    print(out)
if __name__=="__main__":main()
