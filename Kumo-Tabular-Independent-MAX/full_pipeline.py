"""Leakage-aware evaluation pipeline. Kumo SDK inference is validated separately by smoke_predict.py."""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate(df, target, time_col):
    if df.empty or target not in df or time_col not in df:
        raise ValueError("Missing target/time column or empty dataset")
    if df[time_col].isna().any() or df[time_col].duplicated().any():
        raise ValueError("Missing or duplicate timestamps")
    if not df[time_col].is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically")
    if df[target].isna().any() or df[target].nunique() < 2:
        raise ValueError("Target must have at least two classes and no missing labels")
    features = df.drop(columns=[target, time_col])
    if features.empty or not all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise ValueError("Only numeric feature columns supported")
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values")
    return features

def evaluate(df, target="target", time_col="time", min_train=50, step=20):
    X = validate(df, target, time_col)
    y = df[target]
    if min_train < 20 or step < 1 or len(df) <= min_train:
        raise ValueError("Insufficient samples or invalid window")
    rows = []
    for start in range(min_train, len(df), step):
        stop = min(start + step, len(df))
        xtr, xte = X.iloc[:start], X.iloc[start:stop]
        ytr, yte = y.iloc[:start], y.iloc[start:stop]
        if ytr.nunique() < 2:
            continue
        models = {
            "dummy": make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior")),
            "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, random_state=0)),
        }
        try:
            from lightgbm import LGBMClassifier
            models["lightgbm"] = make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(n_estimators=60, verbosity=-1, random_state=0, n_jobs=1))
        except ImportError:
            pass
        for name, model in models.items():
            model.fit(xtr, ytr)
            pred = model.predict(xte)
            item = {"model": name, "train_end": str(df[time_col].iloc[start-1]), "test_start": str(df[time_col].iloc[start]), "test_end": str(df[time_col].iloc[stop-1]), "n": int(stop-start), "accuracy": float(accuracy_score(yte, pred))}
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(xte)
                item["log_loss"] = float(log_loss(yte, prob, labels=model.classes_))
            rows.append(item)
    if not rows:
        raise ValueError("No valid time splits")
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", default="target")
    p.add_argument("--time", default="time")
    p.add_argument("--min-train", type=int, default=50)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--out", default="pipeline_report.json")
    a = p.parse_args()
    data = pd.read_csv(a.csv)
    rows = evaluate(data, a.target, a.time, a.min_train, a.step)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(a.csv), "python": sys.version, "platform": platform.platform(), "rows": len(data), "splits": rows, "note": "Baseline evaluation only; Kumo inference is tested in separate smoke_predict.py"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"success": True, "evaluations": len(rows), "report": str(out)}))

if __name__ == "__main__":
    main()
