"""Fail-closed registry interface for OF-LGBM research candidates.

This module DOES NOT generate features, fit LightGBM, or activate candidates.
Ticket-combination structural scores must never be exposed as pre-draw inputs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REGISTRY_PATH = Path(__file__).with_name("of_lgbm_100_candidate_configs.json")
GROUPS = (
    "frequency_time", "cooccurrence", "repeat_streak_omission",
    "ticket_structure", "openfe_interactions",
)
PRE_DRAW = "pre_draw_feature_proposal"
TICKET = "candidate_ticket_scoring"
ALLOWED_STAGES = frozenset((PRE_DRAW, TICKET))
REQUIRED_FIELDS = frozenset(("id", "group", "name", "stage", "selection", "status"))

def validate_manifest(data):
    """Validate registration and stage semantics; never treat this as proof of no leakage."""
    if type(data) is not dict or data.get("fixed_weights") is not False:
        raise ValueError("registry must disable fixed candidate weights")
    rows = data.get("configurations")
    if type(rows) is not list or len(rows) != 100:
        raise ValueError("expected exactly 100 registered candidates")
    if [r.get("id") for r in rows if isinstance(r, dict)] != [f"{i:03d}" for i in range(1, 101)]:
        raise ValueError("candidate IDs must be complete, unique, and ordered")
    names = []
    counts = Counter()
    for r in rows:
        if not REQUIRED_FIELDS.issubset(r):
            raise ValueError("missing required candidate metadata")
        if not isinstance(r["name"], str) or not r["name"].strip():
            raise ValueError("candidate names must be non-empty")
        if r["group"] not in GROUPS or r["stage"] not in ALLOWED_STAGES:
            raise ValueError("invalid candidate group/stage")
        expected_stage = TICKET if r["group"] == "ticket_structure" else PRE_DRAW
        if r["stage"] != expected_stage:
            raise ValueError("ticket-structure scoring cannot become a pre-draw feature")
        if r["selection"] != "train_fold_only" or r["status"] != "proposed_not_integrated":
            raise ValueError("no candidate can be activated or selected on held-out labels")
        counts[r["group"]] += 1
        names.append(r["name"])
    if len(set(names)) != 100 or counts != Counter({g: 20 for g in GROUPS}):
        raise ValueError("candidate names and group sizes must be unique and balanced")
    return rows

def load_candidates(stage, path=REGISTRY_PATH):
    """Return proposals only for the explicitly requested permitted stage."""
    if stage not in ALLOWED_STAGES:
        raise ValueError("explicit stage is required")
    rows = validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))
    return tuple(dict(r) for r in rows if r["stage"] == stage)
