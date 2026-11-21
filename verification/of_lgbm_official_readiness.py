#!/usr/bin/env python3
"""Fail-closed snapshot integrity and official provenance readiness audit.

A matching checksum or a manifest flag NEVER authenticates an official export.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

GAMES = ('power', 'lotto', '539')


def audit(manifest, snapshot_root=None):
    datasets = manifest.get('datasets', {})
    blockers = []
    checks = {}
    if manifest.get('record_by_record_officially_authenticated') is not True:
        blockers.append('manifest: official record-by-record authentication not confirmed')
    for game in GAMES:
        info = datasets.get(game)
        if not isinstance(info, dict):
            blockers.append(f'{game}: missing dataset metadata')
            continue
        if info.get('official_row_verification') != 'completed':
            blockers.append(f'{game}: official row verification not completed')
        expected_hash = info.get('sha256')
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            blockers.append(f'{game}: valid pinned snapshot SHA-256 missing')
        if game == 'power' and info.get('date_status') == 'missing_all_dates':
            blockers.append('power: 1,964 archived draw dates remain unresolved')
        if snapshot_root is None:
            blockers.append(f'{game}: snapshot bytes have not been checked')
            continue
        rel = info.get('path')
        root = Path(snapshot_root).resolve()
        if not isinstance(rel, str) or not rel.endswith('.csv'):
            blockers.append(f'{game}: snapshot CSV path invalid')
            continue
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            blockers.append(f'{game}: snapshot CSV missing or outside repository')
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        hash_ok = digest == expected_hash
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'draw_id' not in reader.fieldnames:
                blockers.append(f'{game}: snapshot CSV has no draw_id column')
                continue
            count = 0
            dated = 0
            for row in reader:
                count += 1
                if row.get('draw_date'):
                    dated += 1
        checks[game] = {'sha256_matches_manifest': hash_ok,
                        'row_count_matches_manifest': count == info.get('rows'),
                        'rows': count, 'dated_rows': dated}
        if not hash_ok:
            blockers.append(f'{game}: snapshot SHA-256 mismatch')
        if count != info.get('rows'):
            blockers.append(f'{game}: row count differs from manifest')
        if dated != count:
            blockers.append(f'{game}: incomplete draw dates ({dated}/{count})')
    # Even verified snapshot bytes prove only consistency with local metadata.
    # Do not infer real official provenance from a self-described manifest.
    blockers.append('independent official-source authentication is not established by manifest or snapshot checks alone')
    return {'production_official_data_ready': False,
            'snapshot_integrity_checked': snapshot_root is not None,
            'snapshot_checks': checks,
            'official_authentication_performed_by_this_check': False,
            'blockers': blockers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='verification/of_lgbm_data/SOURCE_MANIFEST.json')
    parser.add_argument('--snapshot-root', default='.', help='Repository root containing snapshot files')
    parser.add_argument('--report', help='Optional JSON report path, never overwrites')
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()
    report = audit(json.loads(Path(args.manifest).read_text(encoding='utf-8')),
                   snapshot_root=args.snapshot_root)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.report:
        with Path(args.report).open('x', encoding='utf-8') as f:
            f.write(payload)
    print(payload, end='')
    if args.require_ready and not report['production_official_data_ready']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
