"""Kumo inference on a user-supplied, chronological labeled CSV (CPU/GPU)."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def prepare(df, target, time_col, min_context):
    if target not in df or time_col not in df:
        raise ValueError("Missing target or time column")
    if len(df) <= min_context or min_context < 20:
        raise ValueError("Insufficient context or query rows")
    if df[time_col].isna().any() or df[time_col].duplicated().any() or not df[time_col].is_monotonic_increasing:
        raise ValueError("Time must be non-null, unique and sorted")
    if df[target].isna().any() or df[target].nunique() < 2:
        raise ValueError("Labeled classification data with >=2 classes required")
    features = df.drop(columns=[time_col, target])
    if features.empty:
        raise ValueError("No feature columns")
    if features.columns.duplicated().any():
        raise ValueError("Duplicate features")
    if not all(pd.api.types.is_numeric_dtype(x) for x in features.dtypes):
        raise ValueError("Only numeric features supported by this adapter")
    if features.isna().any().any():
        raise ValueError("Missing features not supported by this adapter")
    return df.drop(columns=[time_col]).copy()

def predict(csv, target="target", time_col="time", min_context=300, out="kumo_csv_report.json"):
    import torch
    import sdm
    from sdm.evaluation import to_class_indices
    from full_pipeline import sha256
    df = pd.read_csv(csv)
    data = prepare(df, target, time_col, min_context)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = sdm.TableTensor.from_pandas(df=data, stypes=sdm.infer_stypes(data, overrides={target:"categorical"}), device=device)
    context, query = table[:min_context], table[min_context:]
    model = sdm.models.KumoTabular(task="classification", size="small", device=device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=device=="cuda"):
        probs = model(x_context=context.drop_columns(target), y_context=context[target], x_query=query.drop_columns(target), num_estimators=8, generator=torch.Generator(device=device).manual_seed(0))
    scores, labels = to_class_indices(probs, query[target])
    if scores.shape[0] != len(df)-min_context:
        raise ValueError("Prediction count mismatch")
    accuracy = float((scores.argmax(dim=-1)==labels).float().mean())
    report = {"success":True, "created_utc":datetime.now(timezone.utc).isoformat(), "source_sha256":sha256(csv), "device":device, "context_rows":min_context, "query_rows":len(df)-min_context, "accuracy":accuracy, "note":"Chronological holdout; not a full walk-forward Kumo benchmark"}
    Path(out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--target",default="target")
    p.add_argument("--time",default="time")
    p.add_argument("--min-context",type=int,default=300)
    p.add_argument("--out",default="kumo_csv_report.json")
    a=p.parse_args()
    print(json.dumps(predict(a.csv,a.target,a.time,a.min_context,a.out),indent=2))
