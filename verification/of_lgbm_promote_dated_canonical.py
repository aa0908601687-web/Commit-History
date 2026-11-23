#!/usr/bin/env python3
"""Install independently cross-checked Power draw dates into the canonical CSV.

This operation is intentionally explicit and fails closed. It does not certify
Taiwan Lottery provenance or override the production attestation release gate.
"""
import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path("verification/of_lgbm_data")
CANON = ROOT / "power_history_archive.csv"
LEGACY = ROOT / "power_history_archive_undated_legacy.csv"
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
DATE_CSV = Path("fresh-official/official-reviewed-Power-dated.csv")
PREV_CSV = Path("prior-stage/power-history-dated-candidate.csv")
REPORT = ROOT / "POWER_DATE_CANONICAL_RECONCILIATION.json"
EXPECTED_SOURCE_HASH = "de92a6019d8efa300f4cbb0e96b7badd9babfea9889b5e7657f0b10a01a80e5b"
EXPECTED_FRESH_HASH = "fc97277d45817f1a673bf2ae02857fc0e7903a871f6ae376c57fcef510d77e68"
ROWS = 1964

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, dated):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        columns={"draw_id","first","second"}|({"draw_date"} if dated else set())
        if not reader.fieldnames or not columns.issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns missing")
        records=[]
        for n,row in enumerate(reader,2):
            draw_id = row["draw_id"].strip()
            if not draw_id.isascii() or not draw_id.isdigit() or len(draw_id)!=9:
                raise ValueError(f"{path}: invalid issue at line {n}")
            vals=tuple(sorted(int(x) for x in row["first"].replace(","," ").split()))
            second=int(row["second"])
            if len(vals)!=6 or len(set(vals))!=6 or any(not 1<=x<=38 for x in vals) or not 1<=second<=8:
                raise ValueError(f"{path}: invalid Power numbers at line {n}")
            day = None
            if dated:
                day = date.fromisoformat(row["draw_date"]).isoformat()
                if int(draw_id[:3]) != date.fromisoformat(day).year-1911:
                    raise ValueError(f"{path}: issue/date ROC mismatch on line {n}")
            records.append((draw_id, day, vals, second))
    if len(records)!=ROWS:
        raise ValueError(f"{path}: expected {ROWS}, got {len(records)}")
    if len(set(r[0] for r in records))!=ROWS:
        raise ValueError(f"{path}: duplicate draw ID")
    if any(int(records[i][0]) >= int(records[i+1][0]) for i in range(ROWS-1)):
        raise ValueError(f"{path}: issue IDs not strictly increasing")
    if dated and any(records[i][1]>=records[i+1][1] for i in range(ROWS-1)):
        raise ValueError(f"{path}: dates not strictly increasing")
    return records

def main():
    if LEGACY.exists() or REPORT.exists():
        raise ValueError("refusing overwrite of existing legacy/archive audit")
    old_raw = CANON.read_bytes()
    fresh_raw = DATE_CSV.read_bytes()
    if sha(old_raw)!=EXPECTED_SOURCE_HASH or sha(fresh_raw)!=EXPECTED_FRESH_HASH:
        raise ValueError("immutable archive or independently fetched source digest differs")
    old=read(CANON,False)
    fresh=read(DATE_CSV,True)
    prior=read(PREV_CSV,True)
    if [(r[0],r[2],r[3]) for r in old]!=[(r[0],r[2],r[3]) for r in fresh]:
        raise ValueError("canonical winning numbers would change; abort")
    if fresh!=prior:
        raise ValueError("independent sources disagree; abort")
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("record_by_record_officially_authenticated") is not False:
        raise ValueError("existing unverified global release flag unexpected")
    p=manifest["datasets"]["power"]
    if p["sha256"]!=EXPECTED_SOURCE_HASH or p["rows"]!=ROWS or p["date_status"]!="missing_all_dates":
        raise ValueError("unexpected existing canonical manifest; abort")
    # Preserve fresh official-source rows (including the newly recovered date column).
    # Do not replace any winning numbers; all rows are checked above.
    shutil.copy2(CANON,LEGACY)
    CANON.write_bytes(fresh_raw)
    p["sha256"]=sha(fresh_raw)
    p["source"]="Taiwan Lottery original annual ZIP CDN, re-downloaded and cross-checked with official API supplement; independent release approval pending"
    p["date_status"]="all_1964_dates_restored_original_source_crosschecked"
    p["official_row_verification"]="cross_checked_not_release_approved"
    p["first"]="097000001"
    p["first_date"]="2008-01-24"
    p["last_date"]="2026-11-16"
    p["source_audit_run"]="https://github.com/aa0908601687-web/Commit-History/actions/runs/45214322634"
    p["source_comparison_run"]="https://github.com/aa0908601687-web/Commit-History/actions/runs/45215425965"
    p["legacy_undated_archive_path"]=str(LEGACY)
    p["legacy_undated_archive_sha256"]=sha(old_raw)
    manifest["power_date_backfill_20261124"]={
       "rows":ROWS,
       "missing_dates":0,
       "source_sha256":sha(fresh_raw),
       "semantic_match_with_prior_candidate":True,
       "official_source_independent_recheck_complete":True,
       "official_source_release_approval":False,
       "production_release_certified":False,
       "note":"Power primary archive now contains recovered ISO dates. Official provenance release gate stays blocked for all games."
    }
    manifest["integrity_audit_20261119"]["notes"]="Historical 2026-11-19 snapshot: dates were missing then. See power_date_backfill_20261124 for subsequent date restoration; official release remains blocked."
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report={
        "status":"CANONICAL_DATES_RESTORED_PENDING_OFFICIAL_RELEASE_APPROVAL",
        "rows":ROWS,"missing_dates":0,
        "numbers_preserved":True,
        "source_semantic_match":True,
        "first_draw_id":fresh[0][0],"last_draw_id":fresh[-1][0],
        "first_date":fresh[0][1],"last_date":fresh[-1][1],
        "legacy_sha256":sha(old_raw),"new_canonical_sha256":sha(fresh_raw),
        "official_release_approved":False,
        "production_official_certification":False
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))

if __name__=="__main__":
    main()
