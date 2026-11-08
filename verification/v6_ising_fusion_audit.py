"""Read-only V6 fused-score -> Ising/QUBO -> exact E2E audit."""
import json
import math
import sys
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
from v6_ising_qubo_reference import build_qubo, energy, exact_six, check_solver_output

def audit(payload):
    f = payload["openfe_feature_layer"]
    if f["engine"] != "OpenFE+LightGBM+PyTorch" or f["pytorch_layer"]["trained"] is not True:
        raise ValueError("trained fusion required")
    weights = f["fusion_weights"]
    if weights != {"lightgbm": 0.5, "pytorch": 0.5}:
        raise ValueError("unexpected fusion weights")
    fused = {int(k): float(v) for k,v in f["fused_scores"].items()}
    light = {int(k): float(v) for k,v in f["lightgbm_scores"].items()}
    torch = {int(k): float(v) for k,v in f["pytorch_scores"].items()}
    if set(fused) != set(light) or set(light) != set(torch) or set(fused) != set(range(1,39)):
        raise ValueError("incomplete score mapping")
    if any(not all(math.isfinite(v) and 0 <= v <= 1 for v in (fused[n], light[n], torch[n])) or
           not math.isclose(fused[n], 0.5*(light[n]+torch[n]), abs_tol=1e-12)
           for n in fused):
        raise ValueError("fusion mismatch")
    pool = sorted(fused, key=lambda n: (-fused[n], n))[:8]
    if pool != payload["candidate_pool"]:
        raise ValueError("candidate mapping mismatch")
    scores = [fused[n] for n in pool]
    result = exact_six(scores)
    chosen = [pool[i] for i in result["indices"]]
    existing = payload["exact_optimization_layer"]
    if sorted(chosen) != sorted(existing["exact_selection"]):
        raise ValueError("Ising and existing exact-search selection differ")
    if not math.isclose(sum(fused[n] for n in chosen), existing["exact_score"], abs_tol=1e-8):
        raise ValueError("objective mismatch")
    bits = [int(i in result["indices"]) for i in range(8)]
    verified = check_solver_output(bits, scores)
    if not verified["optimal"] or result["combinations_checked"] != 28:
        raise ValueError("Ising reference failed")
    cuda = payload["cudaq_layer"]
    if not (cuda["executed"] and cuda["backend"] == "qpp-cpu" and not cuda["real_qpu_connected"]):
        raise ValueError("mandatory local CUDA-Q missing")
    return {"status":"PASS", "pool":pool, "ising_selected":chosen,
            "exact_score":existing["exact_score"], "ising_energy":result["energy"],
            "combinations_checked":28, "cudaq_executed":True,
            "nvidia_native_solver_executed":False, "rocq_proof_verified":False,
            "production_unchanged":True, "predictive_advantage_verified":False}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python verification/v6_ising_fusion_audit.py input.json output.json")
    report = audit(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
