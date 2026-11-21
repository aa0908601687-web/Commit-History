#!/usr/bin/env python3
"""Stage normalized three-lottery history without overwriting pinned source archives.

Input CSVs must be separately acquired and reviewed; passing this program never
certifies that a claimed official source is actually authentic or current.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from of_lgbm_real_history_gate import audit_csv

SCHEMAS = {
    "power": ("power.csv", "power_history_archive.csv", "second"),
    "lotto": ("lotto.csv", "lotto_history_mirror.csv", "special"),
    "539": ("539.csv", "539_history_mirror.csv", None),
}

def rows_by_id(path, game, *, require_date):
    name = SCHEMAS[game][2]
    required = {"draw_id", "first"} | ({"draw_date"} if require_date else set())
    if name:
        required.add(name)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(game + ": missing required columns")
        rows = {}
        for row in reader:
            key = row["draw_id"].strip()
            if key in rows:
                raise ValueError(game + ": duplicate ID " + key)
            first = tuple(sorted(int(v) for v in row["first"].replace(",", " ").split()))
            rows[key] = {
                "first": first,
                "extra": int(row[name]) if name else None,
                "date": row.get("draw_date") or None,
            }
    return rows

def check_all(official_dir, snapshot_dir):
    official_dir, snapshot_dir = Path(official_dir), Path(snapshot_dir)
    source_files = {}
    summary = {}
    for game, (normalized, snapshot, _) in SCHEMAS.items():
        official_path = official_dir / normalized
        snapshot_path = snapshot_dir / snapshot
        if not official_path.is_file():
            raise ValueError(game + ": independent normalized source CSV missing")
        source_audit = audit_csv(official_path, game, min_rows=17)
        local_audit = audit_csv(snapshot_path, game, min_rows=17)
        if source_audit["date_coverage"] != source_audit["rows"]:
            raise ValueError(game + ": source dates incomplete")
        official_rows = rows_by_id(official_path, game, require_date=True)
        local_rows = rows_by_id(snapshot_path, game, require_date=False)
        missing = set(local_rows) - set(official_rows)
        if missing:
            raise ValueError(game + ": source does not cover snapshot IDs (" + str(len(missing)) + " missing)")
        for draw_id, local in local_rows.items():
            external = official_rows[draw_id]
            if local["first"] != external["first"] or local["extra"] != external["extra"]:
                raise ValueError(game + ": draw number mismatch at " + draw_id)
            if local["date"] and local["date"] != external["date"]:
                raise ValueError(game + ": date mismatch at " + draw_id)
        summary[game] = {
            "staged_rows": source_audit["rows"],
            "source_first_id": source_audit["first_id"],
            "source_last_id": source_audit["last_id"],
            "overlap_verified": len(local_rows),
            "previously_undated_snapshot_rows": local_audit["rows"] - local_audit["date_coverage"],
            "sha256": source_audit["sha256"],
            "structural_and_overlap_checks_passed": True,
            "independent_official_provenance_verified": False,
            "complete_to_today_verified": False,
        }
        source_files[game] = official_path
    return source_files, summary

def stage(official_dir, snapshot_dir, output_dir):
    official_dir, snapshot_dir, output_dir = map(Path, (official_dir, snapshot_dir, output_dir))
    destination = output_dir.resolve()
    if destination == snapshot_dir.resolve() or destination == official_dir.resolve():
        raise ValueError("stage directory must be separate")
    if snapshot_dir.resolve() in destination.parents or official_dir.resolve() in destination.parents:
        raise ValueError("stage directory must be outside original source trees")
    if output_dir.exists():
        raise ValueError("stage destination exists; refusing overwrite")
    files, summary = check_all(official_dir, snapshot_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".of-lgbm-stage-", dir=output_dir.parent))
    try:
        for game, original in files.items():
            shutil.copyfile(original, scratch / SCHEMAS[game][0])
            staged_hash = hashlib.sha256((scratch / SCHEMAS[game][0]).read_bytes()).hexdigest()
            if staged_hash != summary[game]["sha256"]:
                raise ValueError("staged file hash mismatch: " + game)
        evidence = {
            "kind": "quarantined_normalized_source_stage_not_main_install",
            "official_provenance_verified": False,
            "full_official_history_verified": False,
            "complete_to_today_verified": False,
            "datasets": summary,
            "required_next_step": "independently authenticate each original official export, assess all-time coverage, and separately approve manifest and main updates",
        }
        (scratch / "STAGING_AUDIT.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(scratch, output_dir)
        return evidence
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, help="Folder containing normalized power.csv, lotto.csv, 539.csv")
    p.add_argument("--snapshot-dir", default="verification/of_lgbm_data")
    p.add_argument("--stage-dir", required=True, help="New, empty destination outside both source directories")
    args = p.parse_args()
    print(json.dumps(stage(args.source_dir, args.snapshot_dir, args.stage_dir), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
