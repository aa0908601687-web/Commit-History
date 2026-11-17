"""Local operational controls: data connectors, drift metrics, manifests and alerts."""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

def ingest(source, destination, timeout=20):
    """Load local CSV or HTTPS CSV. Never embed credentials in source URL."""
    if source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "KumoIndependentMAX/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError("Non-success response")
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError("CSV exceeds 20MB limit")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    elif source.startswith("http://"):
        raise ValueError("HTTPS required for remote sources")
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    df = pd.read_csv(destination)
    if df.empty or df.columns.duplicated().any():
        raise ValueError("Empty CSV or duplicate columns")
    return destination, df

def drift(reference, current, columns, threshold=0.25):
    """Simple standardized mean shift; monitoring heuristic, not a statistical test."""
    report = {}
    for col in columns:
        if col not in reference or col not in current:
            raise ValueError("Missing drift column: " + col)
        a = pd.to_numeric(reference[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            raise ValueError("At least 5 valid values per window required")
        sd = max(float(a.std(ddof=0)), 1e-9)
        shift = abs(float(b.mean() - a.mean())) / sd
        report[col] = {"standardized_mean_shift": shift, "alert": shift > threshold}
    return report

def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def manifest(paths, output, status="completed"):
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "status": status, "files": {}}
    for p in paths:
        f = Path(p)
        if not f.is_file():
            raise FileNotFoundError(f)
        info["files"][str(f)] = {"sha256": fingerprint(f), "size_bytes": f.stat().st_size}
    Path(output).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--source", required=True)
    i.add_argument("--out", required=True)
    d = sub.add_parser("drift")
    d.add_argument("--reference", required=True)
    d.add_argument("--current", required=True)
    d.add_argument("--columns", required=True)
    d.add_argument("--out", default="drift_report.json")
    m = sub.add_parser("manifest")
    m.add_argument("--files", nargs="+", required=True)
    m.add_argument("--out", default="run_manifest.json")
    a = parser.parse_args()
    if a.command == "ingest":
        p, df = ingest(a.source, a.out)
        print(json.dumps({"rows": len(df), "sha256": fingerprint(p)}))
    elif a.command == "drift":
        ref, cur = pd.read_csv(a.reference), pd.read_csv(a.current)
        result = drift(ref, cur, a.columns.split(","))
        Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    else:
        print(json.dumps(manifest(a.files, a.out)))
if __name__ == "__main__":
    main()
