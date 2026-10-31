#!/usr/bin/env python3
"""Execute a bounded runtime probe against selected pinned upstream modules.

This verifies Python runtime interoperability only. It does not execute the
upstream main.py, train a lottery model, or feed WP-OS predictions.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path

PINNED_COMMIT = "da09a58ae32aed50cdf13caa8a5e2f4c795a324d"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(upstream: Path) -> dict:
    upstream = upstream.resolve()
    base_module = _load(
        "pinned_superintelligence_base_agent",
        upstream / "agents" / "base_agent.py",
    )
    learning_module = _load(
        "pinned_superintelligence_adaptive_learning",
        upstream / "auto_improvement" / "adaptive_learning.py",
    )

    class ProbeModel:
        def __init__(self):
            self.calls = []

        def fit(self, data, labels, epochs):
            self.calls.append({"data": data, "labels": labels, "epochs": epochs})

    class ProbeAgent(base_module.BaseAgent):
        def perform_task(self, data):
            return {"agent_id": self.id, "data": data}

    model = ProbeModel()
    learner = learning_module.AdaptiveLearning(model)
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        learner.continuous_learning([[1.0]], [1])
    agent_result = ProbeAgent(model, "runtime-probe").perform_task("ok")

    passed = (
        model.calls == [{"data": [[1.0]], "labels": [1], "epochs": 5}]
        and agent_result == {"agent_id": "runtime-probe", "data": "ok"}
    )
    return {
        "schema_version": 1,
        "upstream_commit": PINNED_COMMIT,
        "runtime_probe_passed": passed,
        "modules_executed": [
            "agents/base_agent.py",
            "auto_improvement/adaptive_learning.py",
        ],
        "fit_call": model.calls[0] if model.calls else None,
        "agent_result": agent_result,
        "upstream_stdout": captured.getvalue().splitlines(),
        "upstream_main_executed": False,
        "tensorflow_automl_executed": False,
        "prediction_integration": False,
        "predictive_advantage_verified": False,
        "note": (
            "Bounded interoperability probe only; broken or dependency-heavy "
            "upstream entry points are deliberately not executed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe(args.upstream_dir)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["runtime_probe_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
