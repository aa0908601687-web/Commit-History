#!/usr/bin/env python3
"""Fail-closed, local-snapshot historical completeness audit for OF-LGBM.

A sequential archive snapshot is NOT an all-time official history certificate.
This report never repairs or authenticates third-party data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from of_lgbm_real_history_gate import audit_csv

FILES = {"power": "power_history_archive.csv",
         "lotto": "lotto_history_mirror.csv", "539": "539_history_mirror.csv"}

def audit_all(data_dir, manifest_path):
    root = Path(data_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if set(manifest.get("datasets", {})) != set(FILES):
        raise ValueError("manifest game list mismatch")
    if manifest.get("record_by_record_officially_authenticated") is not False:
        raise ValueError("unsubstantiated official provenance")
    output = {}
    for game, name in FILES.items():
        item = manifest["datasets"][game]
        path = root / name
        if item["path"] != "verification/of_lgbm_data/" + name:
            raise ValueError("unexpected data path for " + game)
        info = audit_csv(path, game)
        cutoff = item.get("last", item.get("cutoff"))
        if info["sha256"] != item["sha256"]:
            raise ValueError("SHA-256 mismatch for " + game)
        if info["rows"] != item["rows"] or info["last_id"] != cutoff:
            raise ValueError("count or cutoff mismatch for " + game)
        if game == "power":
            if item.get("official_row_verification") != "cross_checked_not_release_approved":
                raise ValueError("unsubstantiated official attestation")
            if manifest.get("power_date_backfill_20261124", {}).get("official_source_release_approval") is not False:
                raise ValueError("unsubstantiated official attestation")
        elif item.get("official_row_verification") != "not_completed":
            raise ValueError("unsubstantiated official attestation")
        if not info["chronological_and_legal_range_verified"]:
            raise ValueError("sequence integrity failed")
        dates_missing = info["rows"] - info["date_coverage"]
        expected_status = ("all_1964_dates_restored_original_source_crosschecked" if game == "power"
                           else "format_consistent_not_officially_verified")
        if item["date_status"] != expected_status:
            raise ValueError("incorrect date provenance marker for " + game)
        if (game == "power" and info["date_coverage"] != info["rows"] or
            game != "power" and info["date_coverage"] != info["rows"]):
            raise ValueError("date coverage contradicts pinned manifest for " + game)
        output[game] = {
            "snapshot_rows": info["rows"],
            "first_id": info["first_id"],
            "last_id": info["last_id"],
            "missing_draw_dates": dates_missing,
            "date_coverage": info["date_coverage"],
            "sha256_matches_manifest": True,
            "ids_legal_sequential_within_snapshot": True,
            "full_historical_coverage_certified": False,
            "official_record_by_record_verified": False,
            "verified_current_to_latest_official_draw": False,
            "date_status": item["date_status"],
            "sha256": info["sha256"],
        }
    if manifest["integrity_audit_20261119"]["missing_power_draw_dates"] != manifest.get("power_date_backfill_20261124",{}).get("rows"):
        raise ValueError("historical Power missing-date tally contradicts original pinned disclosure")
    if output["power"]["missing_draw_dates"] != manifest.get("power_date_backfill_20261124",{}).get("missing_dates"):
        raise ValueError("Power missing-date tally contradicts pinned backfill")
    return {
        "audit_type": "local_snapshot_only_not_official_certification",
        "dataset_count": len(output),
        "official_full_history_verified": False,
        "complete_to_today_verified": False,
        "datasets": output,
        "next_step": "Power draw dates restored and original-source crosschecked; formal provenance release approval and full-history freshness validation remain required"
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="verification/of_lgbm_data")
    parser.add_argument("--manifest", default="verification/of_lgbm_data/SOURCE_MANIFEST.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit_all(args.data_dir, args.manifest)
    body = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(body, encoding="utf-8")
    print(body, end="")

if __name__ == "__main__":
    main()
