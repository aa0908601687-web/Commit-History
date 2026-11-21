#!/usr/bin/env python3
"""Fail-closed check of official data readiness; does not authenticate source files."""
import argparse
import json
from pathlib import Path

GAMES = ('power', 'lotto', '539')

def audit(manifest):
    datasets = manifest.get('datasets', {})
    blockers = []
    if manifest.get('record_by_record_officially_authenticated') is not True:
        blockers.append('manifest: official record-by-record authentication not confirmed')
    for game in GAMES:
        info = datasets.get(game)
        if not isinstance(info, dict):
            blockers.append(f'{game}: missing dataset metadata')
            continue
        if info.get('official_row_verification') != 'completed':
            blockers.append(f'{game}: official row verification not completed')
        if not info.get('sha256'):
            blockers.append(f'{game}: snapshot SHA-256 missing')
        if game == 'power' and info.get('date_status') == 'missing_all_dates':
            blockers.append('power: 1,964 archived draw dates remain unresolved')
    return {'production_official_data_ready': not blockers,
            'official_authentication_performed_by_this_check': False,
            'blockers': blockers}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='verification/of_lgbm_data/SOURCE_MANIFEST.json')
    parser.add_argument('--report', help='Optional JSON report path, never overwrites')
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()
    report = audit(json.loads(Path(args.manifest).read_text(encoding='utf-8')))
    payload = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.report:
        with Path(args.report).open('x', encoding='utf-8') as f:
            f.write(payload)
    print(payload, end='')
    if args.require_ready and not report['production_official_data_ready']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
