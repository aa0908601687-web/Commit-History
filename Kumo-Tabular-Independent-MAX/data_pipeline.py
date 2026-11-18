"""Independent, leakage-safe tabular data preparation and audit helpers."""
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_csv(path, target, time_column=None):
    """Check row shape, nonempty target, duplicate rows and chronological order."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("Missing or duplicate column names")
        if target not in fields:
            raise ValueError("Missing target column")
        if time_column and time_column not in fields:
            raise ValueError("Missing time column")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("At least three data rows required")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    if any(not (row[target] or "").strip() for row in rows):
        raise ValueError("Missing target values")
    fingerprints = [tuple(row[c] for c in fields) for row in rows]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Duplicate data rows")
    if time_column:
        dates = [row[time_column] for row in rows]
        if any(not v.strip() for v in dates) or dates != sorted(dates):
            raise ValueError("Time column must be nonempty and ascending ISO-sortable")
    return {"fields": fields, "rows": rows, "row_count": len(rows),
            "sha256": sha256_file(path)}

def chronological_split(rows, fraction=0.8):
    """Strictly ordered split; shuffle is forbidden for time series."""
    if not 0 < fraction < 1:
        raise ValueError("Invalid split fraction")
    boundary = max(1, min(len(rows)-1, int(len(rows)*fraction)))
    return rows[:boundary], rows[boundary:]

def evaluate_classification(y_true, y_pred):
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Prediction/label mismatch")
    return {"count": len(y_true),
            "accuracy": sum(a == b for a, b in zip(y_true, y_pred))/len(y_true)}

def evaluate_regression(y_true, y_pred):
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Prediction/label mismatch")
    pairs = [(float(a), float(b)) for a,b in zip(y_true,y_pred)]
    if not all(math.isfinite(a) and math.isfinite(b) for a,b in pairs):
        raise ValueError("Nonfinite prediction or label")
    return {"count": len(pairs), "mae": sum(abs(a-b) for a,b in pairs)/len(pairs),
            "rmse": math.sqrt(sum((a-b)**2 for a,b in pairs)/len(pairs))}

def write_audit(path, source, meta):
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha256_file(source), "metadata": meta}
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["report_sha256"] = sha256_file(path)
    return result
