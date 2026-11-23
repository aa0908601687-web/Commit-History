#!/usr/bin/env python3
"""Check source-review evidence is pinned and cannot authorize official release."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
META=ROOT/"verification/of_lgbm_data/SOURCE_MANIFEST.json"
REVIEW=ROOT/"verification/of_lgbm_data/POWER_SOURCE_INTERNAL_REVIEW_20261124.json"

def audit():
    manifest=json.loads(META.read_text(encoding="utf-8"))
    review=json.loads(REVIEW.read_text(encoding="utf-8"))
    p=manifest["datasets"]["power"]
    snapshot=ROOT/p["path"]
    raw=snapshot.read_bytes()
    assert review["decision"]=="APPROVED_FOR_INTERNAL_RESEARCH_SNAPSHOT_USE"
    assert p["internal_source_review_status"]=="approved_for_research_snapshot_only"
    assert p["internal_source_review_record"]==str(REVIEW.relative_to(ROOT))
    assert p["research_snapshot_use_approved"] is True
    assert p["production_use_approved"] is False
    assert manifest["record_by_record_officially_authenticated"] is False
    assert manifest["power_date_backfill_20261124"]["official_source_release_approval"] is False
    assert review["restrictions"]["production_release_approved"] is False
    assert review["restrictions"]["taiwan_lottery_official_endorsement_or_certification"] is False
    assert review["restrictions"]["all_three_games_officially_authenticated"] is False
    assert p["official_row_verification"]=="cross_checked_not_release_approved"
    assert len(review["evidence"])>=4
    assert hashlib.sha256(raw).hexdigest()==p["sha256"]==review["scope"]["sha256"]
    with snapshot.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1964==review["scope"]["records"]
    assert all(r["draw_date"] for r in rows)
    assert rows[0]["draw_id"]==review["scope"]["start_issue"]
    assert rows[-1]["draw_id"]==review["scope"]["end_issue"]
    return {"research_snapshot_accepted":True,"records":len(rows),
            "official_release_approved":False,"production_deployment":False}

if __name__=="__main__":
    print(json.dumps(audit(),ensure_ascii=False))
