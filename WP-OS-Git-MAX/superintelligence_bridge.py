#!/usr/bin/env python3
"""Controlled bridge to the pinned Superintelligence architecture reference.

This bridge deliberately does not import or execute upstream code. It exposes a
small, deterministic contract for turning an existing pipeline report into
monitoring recommendations. It has zero prediction weight and cannot alter a
lottery selection.
"""
import hashlib
import json

BRIDGE_VERSION = "superintelligence-reference-bridge-v1"
UPSTREAM_REPOSITORY = "https://github.com/marciolscoutinho/superintelligence"
UPSTREAM_COMMIT = "da09a58ae32aed50cdf13caa8a5e2f4c795a324d"


def analyze(pipeline_report):
    required = {"pipeline_version", "through_draw", "predictions", "sha256"}
    missing = sorted(required - set(pipeline_report))
    if missing:
        raise ValueError("Missing pipeline fields: " + ", ".join(missing))

    predictions = pipeline_report["predictions"]
    if not predictions:
        raise ValueError("At least one prediction report is required")
    cutoffs = {item.get("through_draw") for item in predictions.values()}
    if cutoffs != {pipeline_report["through_draw"]}:
        raise ValueError("Prediction cutoff mismatch")

    rejected = sorted(
        name for name, status in pipeline_report.get("feature_checks", {})
        .get("models", {}).items() if not status.get("accepted", False)
    )
    overlaps = pipeline_report.get("first_zone_overlap", {})
    sparse_agreement = sorted(name for name, values in overlaps.items() if len(values) < 2)
    recommendations = []
    if rejected:
        recommendations.append("review_feature_check_rejections")
    if sparse_agreement:
        recommendations.append("monitor_low_model_agreement")
    if not recommendations:
        recommendations.append("no_diagnostic_alert")

    result = {
        "bridge_version": BRIDGE_VERSION,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        "through_draw": pipeline_report["through_draw"],
        "input_pipeline_sha256": pipeline_report["sha256"],
        "mode": "diagnostic_reference_only",
        "prediction_weight": 0.0,
        "selection_mutation_allowed": False,
        "upstream_code_executed": False,
        "rejected_models": rejected,
        "sparse_agreement_pairs": sparse_agreement,
        "recommendations": recommendations,
        "claim_boundary": "Architecture reference only; no predictive advantage is claimed.",
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result
