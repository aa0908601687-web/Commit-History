#!/usr/bin/env python3
"""Real, opt-in Chronos-2 and t0-gamma inference on 20-draw binary indicator series.

Requires actual model weight downloads and compatible runtimes. Never substitutes
synthetic predictions; failure is fatal. No claim of lottery predictability.
"""
import argparse
import json
import math
from pathlib import Path

from wp_os_git_max import load

def contexts(draws):
    if len(draws) < 20:
        raise ValueError("At least 20 completed draws required")
    window = draws[-20:]
    return [[float((n in d["first"]) if n <= 38 else (d["second"] == n - 38))
             for d in window] for n in range(1, 47)]

def score_map(values):
    if len(values) != 46 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Model did not return 46 finite forecasts")
    # A positive finite transform of regression forecasts; NOT calibrated probabilities.
    probs = [1.0 / (1.0 + math.exp(-max(-30, min(30, float(x))))) for x in values]
    return {
        "first": {str(i): probs[i - 1] for i in range(1,39)},
        "second": {str(i): probs[38 + i - 1] for i in range(1,9)}
    }

def infer_chronos(rows):
    import torch
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    inputs = torch.tensor(rows, dtype=torch.float32)
    quantiles, means = pipeline.predict_quantiles(inputs, prediction_length=1,
                                                   quantile_levels=[0.5])
    # Implementation may return one tensor for a batched 2D array.
    predictions = torch.as_tensor(means[0]).reshape(-1) if len(means) == 1 else torch.stack([
        torch.as_tensor(item).reshape(-1)[0] for item in means])
    if predictions.numel() != 46:
        raise ValueError("Chronos-2 forecast must contain 46 channels")
    return score_map(predictions.tolist())

def infer_t0(rows):
    import numpy as np
    from t0 import T0Forecaster
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").eval()
    output = model.predict(np.asarray(rows, dtype=np.float32), horizon=1,
                           quantile_levels=[0.5])
    med = output.median
    med = med.detach().cpu().numpy() if hasattr(med, "detach") else np.asarray(med)
    return score_map(med.reshape(-1).tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    draws = load(args.csv)
    rows = contexts(draws)
    # Both calls must succeed; no silent fallback or synthetic fill.
    models = {"chronos_2": infer_chronos(rows), "t0_gamma": infer_t0(rows)}
    result = {"through_draw": draws[-1]["draw_id"], "models": models,
              "execution_mode": "real_pretrained_inference",
              "note": "Regression output converted to heuristic scores; not calibrated lottery odds."}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
