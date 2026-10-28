#!/usr/bin/env python3
"""Fail-closed ABE five-model end-to-end acceptance gate.

Run after preparing a verified historical CSV with >=20 completed draws.
Set ABE_LSTM_CMD, ABE_GRU_CMD, ABE_CHRONOS2_CMD, ABE_T0_GAMMA_CMD
to executable commands that write ONE JSON prediction to stdout.
Each command receives --csv <path> appended to its arguments.
JSON must include model, first (six 1..38 ints), second (1..8 int),
through_draw, backend, checkpoint. Placeholder/stub backends are rejected.
This gate establishes execution and provenance, NOT predictive accuracy.
"""
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "historical_archive.csv"
MODELS = {
    "bayesian": [sys.executable, str(ROOT / "adaptive_bayesian_ensemble_max.py")],
    "lstm": os.getenv("ABE_LSTM_CMD", ""),
    "gru": os.getenv("ABE_GRU_CMD", ""),
    "chronos-2": os.getenv("ABE_CHRONOS2_CMD", ""),
    "t0-gamma": os.getenv("ABE_T0_GAMMA_CMD", ""),
}
INVALID_BACKENDS = {"", "stub", "mock", "placeholder", "synthetic", "dummy"}


def fail(message):
    raise SystemExit("ABE E2E FAILED: " + message)


def main():
    if not CSV.is_file():
        fail("historical archive missing")
    with CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 20:
        fail("requires >=20 completed historical draws")
    target = rows[-1]["draw_id"]
    if not target:
        fail("missing last draw ID")
    reports = {}
    for name, cmd in MODELS.items():
        if not cmd:
            fail(name + " inference command not configured")
        argv = cmd if isinstance(cmd, list) else shlex.split(cmd)
        try:
            p = subprocess.run(argv + ["--csv", str(CSV)], text=True,
                               capture_output=True, check=True, timeout=180)
            data = json.loads(p.stdout)
        except (OSError, ValueError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            fail(name + " true inference failed: " + str(exc))
        if name != "bayesian":
            if str(data.get("model", "")).strip().lower() != name:
                fail(name + " model identity mismatch")
            if str(data.get("backend", "")).strip().lower() in INVALID_BACKENDS:
                fail(name + " has no verified execution backend")
            if str(data.get("checkpoint", "")).strip().lower() in INVALID_BACKENDS:
                fail(name + " has no checkpoint/version provenance")
        first = data.get("first")
        second = data.get("second")
        if (not isinstance(first, list) or len(first) != 6
                or any(type(n) is not int or not 1 <= n <= 38 for n in first)
                or len(set(first)) != 6):
            fail(name + " invalid first-zone prediction")
        if type(second) is not int or not 1 <= second <= 8:
            fail(name + " invalid second-zone prediction")
        if str(data.get("through_draw")) != target:
            fail(name + " stale/incorrect draw cutoff")
        reports[name] = {"through_draw": target, "first": first, "second": second}
    print(json.dumps({"status": "E2E inference contracts passed",
                      "models": list(reports), "through_draw": target},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
