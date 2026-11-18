"""Audit the complete research pipeline scope without claiming unimplemented functions work.

This is a static repository capability audit. Real inference is tested separately in e2e.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checks = [
    ("01 Python runtime", True, "GitHub Actions uses Python 3.11"),
    ("02 Official local SDK install", (ROOT/"requirements.txt").is_file(), "Install job runs pip and pip check"),
    ("03 CSV data ingestion", (ROOT/"data_ingest.py").is_file(), "Dedicated, validated CSV ingestion is required"),
    ("04 Data quality validation", (ROOT/"data_quality.py").is_file(), "Missing-value, duplicate, and schema validation required"),
    ("05 Data preprocessing", (ROOT/"preprocess.py").is_file(), "Reusable preprocessing pipeline required"),
    ("06 Feature and label setup", (ROOT/"features.py").is_file(), "Reusable feature/label mapping required"),
    ("07 Leakage-free chronological splits", (ROOT/"time_split.py").is_file(), "Time-based split validation required"),
    ("08 Actual KumoTabular load", (ROOT/"smoke_predict.py").is_file(), "Checked in real model E2E job"),
    ("09 Actual model inference", (ROOT/"smoke_predict.py").is_file(), "Checked in real model E2E job"),
    ("10 Results report", (ROOT/"smoke_predict.py").is_file(), "Inference smoke JSON output"),
    ("11 Walk-forward backtest", (ROOT/"walk_forward.py").is_file(), "Sequential historical evaluation is required"),
    ("12 Random/baseline comparison", (ROOT/"baselines.py").is_file(), "Fair baseline comparison required"),
    ("13 Unit tests", (ROOT/"tests/test_package.py").is_file(), "Checked in unit job"),
    ("14 Integrated E2E", (ROOT/"full_pipeline_e2e.py").is_file(), "Full input-to-evaluation test required; smoke E2E exists separately"),
    ("15 GitHub CI automation", True, "Dedicated Kumo workflow exists"),
]
report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "audit_type": "static_capability_presence_not_execution_proof",
    "capabilities": [
        {"stage": name, "present": present, "evidence_or_gap": explanation}
        for name, present, explanation in checks
    ],
    "present_count": sum(bool(p) for _, p, _ in checks),
    "total_count": len(checks),
    "full_research_e2e_verified": False,
    "notes": "Existing unit/install/model-inference E2E job status is checked separately in GitHub Actions. Presence alone does not imply implementation quality or successful end-to-end tests."
}
(ROOT/"full_scope_audit_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))
