#!/usr/bin/env python3
"""V6 device-consistency smoke test. Not a trained predictive model."""
import argparse
import json
from pathlib import Path
from v6_exact_search import exact_select

POOL = list(range(1, 9))

def compute_scores(torch, device):
    x = torch.arange(1, 9, dtype=torch.float64, device=device)
    values = x.square() / 17.0 + x / 13.0
    return dict(zip(POOL, map(float, values.cpu().tolist())))

def verify(torch, require_gpu=False, tolerance=1e-8):
    cpu_scores = compute_scores(torch, "cpu")
    selected, score, count = exact_select(POOL, cpu_scores)
    assert count == 28
    report = {
        "scope": "CPU/GPU deterministic smoke test; not predictive-model validation",
        "torch_version": torch.__version__,
        "cpu": {"status": "PASS", "device": "cpu", "selection": selected, "score": score,
                "combinations": count},
        "gpu": {"status": "NOT_TESTED", "reason": "CUDA device unavailable"},
        "predictive_advantage_verified": False,
    }
    available = bool(torch.cuda.is_available())
    if require_gpu and not available:
        raise RuntimeError("GPU verification requested but CUDA is unavailable")
    if available:
        gpu_scores = compute_scores(torch, "cuda:0")
        gpu_selected, gpu_score, gpu_count = exact_select(POOL, gpu_scores)
        gap = max(abs(cpu_scores[n] - gpu_scores[n]) for n in POOL)
        ok = gap <= tolerance and gpu_selected == selected and gpu_count == 28
        report["gpu"] = {
            "status": "PASS" if ok else "FAIL",
            "device": torch.cuda.get_device_name(0),
            "selection": gpu_selected,
            "score": gpu_score,
            "combinations": gpu_count,
            "same_selection": gpu_selected == selected,
            "max_score_gap": gap,
            "tolerance": tolerance,
        }
        if not ok:
            raise AssertionError("CPU/GPU result differs beyond tolerance")
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--output", default="v6-cpu-gpu-evidence.json")
    a = p.parse_args()
    import torch
    result = verify(torch, require_gpu=a.require_gpu)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
