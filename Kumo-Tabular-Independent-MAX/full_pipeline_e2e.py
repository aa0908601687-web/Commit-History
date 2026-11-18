"""Isolated research pipeline E2E on deterministic synthetic chronological CSV.

This validates *actual* KumoTabular inference with chronology-preserving data
and a comparable baseline. No claim of real-world/l lottery performance.
"""
import csv
import hashlib
import json
import math
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import sdm
from sdm.evaluation import to_class_indices

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research_pipeline_report.json"

def make_demo_csv(path, n=90):
    rng = random.Random(20261118)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "signal", "noise", "target"])
        writer.writeheader()
        for i in range(n):
            signal = rng.uniform(-1, 1)
            noise = rng.uniform(-1, 1)
            target = int(signal + 0.2 * noise + rng.uniform(-0.45, 0.45) > 0)
            writer.writerow({"timestamp": f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00", "signal": signal, "noise": noise, "target": target})

def load_and_validate(path):
    frame = pd.read_csv(path)
    cols = ["timestamp", "signal", "noise", "target"]
    if list(frame.columns) != cols or len(frame) < 70:
        raise ValueError("Missing or misordered columns, or insufficient rows")
    if frame.isna().any().any() or frame["timestamp"].duplicated().any():
        raise ValueError("Missing values or duplicate timestamps")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Data must already be chronologically ordered")
    for name in ("signal", "noise"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not all(math.isfinite(float(v)) for v in frame[name]):
            raise ValueError("Non-finite feature values")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
    if not set(frame["target"]).issubset({0, 1}):
        raise ValueError("Target must contain binary labels")
    return frame

def folds(n):
    for split in (50, 65, 80):
        test_end = min(split + 10, n)
        if test_end <= split:
            break
        yield split, test_end

def run():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "synthetic_chronological.csv"
        make_demo_csv(path)
        frame = load_and_validate(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sdm.models.KumoTabular(task="classification", size="small", device=device)
    rows = []
    for train_end, test_end in folds(len(frame)):
        train = frame.iloc[:train_end]
        holdout = frame.iloc[train_end:test_end]
        assert train["timestamp"].max() < holdout["timestamp"].min()
        frame_for_sdm = frame.iloc[:test_end][["signal", "noise", "target"]].copy()
        table = sdm.TableTensor.from_pandas(
            df=frame_for_sdm,
            stypes=sdm.infer_stypes(frame_for_sdm, overrides={"target": "categorical"}),
            device=device,
        )
        context, query = table[:train_end], table[train_end:test_end]
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device, dtype=torch.float16, enabled=device == "cuda"
        ):
            probs = model(
                x_context=context.drop_columns("target"),
                y_context=context["target"],
                x_query=query.drop_columns("target"),
                num_estimators=4,
                generator=torch.Generator(device=device).manual_seed(20261118),
            )
        scores, labels = to_class_indices(probs, query["target"])
        predicted = scores.argmax(dim=-1)
        model_accuracy = float((predicted == labels).float().mean())
        baseline_label = int(train["target"].mean() >= 0.5)
        baseline_accuracy = float((holdout["target"].to_numpy() == baseline_label).mean())
        assert len(scores) == len(holdout) and math.isfinite(model_accuracy)
        assert 0 <= model_accuracy <= 1 and 0 <= baseline_accuracy <= 1
        rows.append({
            "train_rows": train_end,
            "test_rows": len(holdout),
            "first_test_timestamp": holdout["timestamp"].min().isoformat(),
            "last_train_timestamp": train["timestamp"].max().isoformat(),
            "kumo_accuracy": model_accuracy,
            "majority_baseline_accuracy": baseline_accuracy,
        })
    assert len(rows) == 3
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "real_kumo_inference_synthetic_chronological_demo",
        "data_provenance": "deterministic locally generated synthetic CSV; not real-world/l lottery data",
        "device": device,
        "data_sha256": digest,
        "rows": len(frame),
        "model": "sdm.models.KumoTabular classification small",
        "walk_forward_folds": rows,
        "mean_kumo_accuracy": sum(x["kumo_accuracy"] for x in rows)/len(rows),
        "mean_majority_baseline_accuracy": sum(x["majority_baseline_accuracy"] for x in rows)/len(rows),
        "passed": True,
        "limitations": [
            "Synthetic ordered test data cannot establish real-world accuracy or predictive advantage",
            "No externally sourced production data ingestion or deployment tested",
            "Three origin windows are engineering E2E smoke checks, not statistical evidence of performance"
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run()
