#!/usr/bin/env python3
"""Read-only fail-closed production preflight, NOT official certification.

Report project readiness blockers and never request or perform deployment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "verification/of_lgbm_data/SOURCE_MANIFEST.json"
REVIEW = ROOT / "verification/of_lgbm_data/POWER_SOURCE_INTERNAL_REVIEW_20261124.json"

def audit():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    power=manifest["datasets"]["power"]
    review=json.loads(REVIEW.read_text(encoding="utf-8"))
    target=ROOT / power["path"]
    raw=target.read_bytes()
    with target.open(encoding="utf-8-sig", newline="") as handle:
        records=list(csv.DictReader(handle))
    dates_present=all(r.get("draw_date","").strip() for r in records)
    checksum_verified=(hashlib.sha256(raw).hexdigest()==power["sha256"]==
                       review["scope"]["sha256"])
    matches_reference=(len(records)==1964 and records[0]["draw_id"]=="097000001" and
                       records[-1]["draw_id"]=="115000092" and dates_present)
    blockers=[]
    if not checksum_verified:
        blockers.append("power: primary CSV checksum disagrees with source review")
    if not matches_reference:
        blockers.append("power: primary historical dates or coverage disagree with approved research snapshot")
    if manifest.get("record_by_record_officially_authenticated") is not True:
        blockers.append("official authentication: global three-game official attestation absent")
    for name,details in manifest["datasets"].items():
        if details.get("official_row_verification")!="completed":
            blockers.append(f"{name}: official source final release approval absent")
    if power.get("production_use_approved") is not True:
        blockers.append("power: research-only acceptance does not authorize production use")
    if review["restrictions"].get("production_release_approved") is not True:
        blockers.append("power: source-review evidence explicitly blocks production release")
    blockers.append("deployment: no independently verified production database destination, protected environment approval, rollback evidence or post-deploy E2E provided to this check")
    blockers.append("freshness: snapshot ends on 2026-11-16; complete current historical coverage not attested")
    return {
      "check":"OF_LGBM_PRODUCTION_RELEASE_PREFLIGHT",
      "production_ready":False,
      "official_certification_granted_by_this_check":False,
      "deployment_attempted":False,
      "power_research_snapshot_rows":len(records),
      "power_research_snapshot_all_dates_present":dates_present,
      "power_research_snapshot_sha256_valid":checksum_verified,
      "power_snapshot_cutoff":power["cutoff"],
      "blockers":blockers
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing overwrite")
    report=audit()
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    if report["production_ready"]:
        raise SystemExit("unexpected unsafe readiness assertion")
if __name__=="__main__":
    main()
