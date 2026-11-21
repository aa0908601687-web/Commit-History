#!/usr/bin/env python3
"""Fail-closed snapshot integrity and official provenance readiness audit.

A matching checksum or a manifest flag NEVER authenticates an official export.
"""
import argparse
import csv
import hashlib
import json
from datetime import date
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
            malformed_dates = 0
            duplicate_ids = 0
            ids = set()
            first_id = last_id = None
            previous_num = None
            previous_year = previous_sequence = None
            missing_sequence_events = 0
            invalid_year_rollovers = 0
            previous_date = None
            malformed_ids = 0
            unordered_ids = 0
            unordered_dates = 0
            date_year_mismatches = 0
            invalid_number_rows = 0
            max_number, expected_count, extra_key = {
                'power': (38, 6, 'second'),
                'lotto': (49, 6, 'special'),
                '539': (39, 5, None),
            }[game]
            required = {'draw_id', 'first'} | ({extra_key} if extra_key else set())
            if not required.issubset(reader.fieldnames):
                blockers.append(f'{game}: missing winning-number CSV columns')
                continue
            for row in reader:
                count += 1
                try:
                    numbers = [int(n) for n in (row.get('first') or '').replace(',', ' ').split()]
                    valid = (len(numbers) == expected_count
                             and len(set(numbers)) == expected_count
                             and all(1 <= n <= max_number for n in numbers))
                    if extra_key:
                        extra = int(row.get(extra_key) or '')
                        valid = valid and (1 <= extra <= (8 if game == 'power' else 49))
                        if game == 'lotto':
                            valid = valid and extra not in numbers
                    if not valid:
                        invalid_number_rows += 1
                except (ValueError, TypeError, AttributeError):
                    invalid_number_rows += 1
                draw_id = (row.get('draw_id') or '').strip()
                if not draw_id or draw_id in ids:
                    duplicate_ids += 1
                ids.add(draw_id)
                if len(draw_id) != 9 or not draw_id.isascii() or not draw_id.isdigit():
                    malformed_ids += 1
                else:
                    current_num = int(draw_id)
                    if previous_num is not None and current_num <= previous_num:
                        unordered_ids += 1
                    year, sequence = divmod(current_num, 1000000)
                    if year < 1 or sequence < 1:
                        malformed_ids += 1
                    elif previous_year is not None:
                        if year == previous_year and sequence > previous_sequence + 1:
                            missing_sequence_events += 1
                        elif year != previous_year and (
                            year != previous_year + 1 or sequence != 1
                        ):
                            invalid_year_rollovers += 1
                    previous_num = current_num
                    previous_year, previous_sequence = year, sequence
                if first_id is None:
                    first_id = draw_id
                last_id = draw_id
                value = (row.get('draw_date') or '').strip()
                if value:
                    dated += 1
                    try:
                        parsed = date.fromisoformat(value)
                        if parsed.isoformat() != value:
                            malformed_dates += 1
                        if previous_date is not None and parsed <= previous_date:
                            unordered_dates += 1
                        previous_date = parsed
                        if len(draw_id) == 9 and draw_id.isdigit() and parsed.year - 1911 != int(draw_id[:3]):
                            date_year_mismatches += 1
                    except ValueError:
                        malformed_dates += 1
        checks[game] = {'sha256_matches_manifest': hash_ok,
                        'row_count_matches_manifest': count == info.get('rows'),
                        'rows': count, 'dated_rows': dated,
                        'invalid_or_duplicate_ids': duplicate_ids,
                        'malformed_dates': malformed_dates,
                        'malformed_ids': malformed_ids,
                        'unordered_ids': unordered_ids,
                        'missing_sequence_events': missing_sequence_events,
                        'invalid_year_rollovers': invalid_year_rollovers,
                        'unordered_dates': unordered_dates,
                        'date_year_mismatches': date_year_mismatches,
                        'invalid_number_rows': invalid_number_rows}
        if duplicate_ids:
            blockers.append(f'{game}: empty or duplicate draw IDs ({duplicate_ids})')
        if invalid_number_rows:
            blockers.append(f'{game}: invalid winning-number rows ({invalid_number_rows})')
        if malformed_dates:
            blockers.append(f'{game}: invalid ISO dates ({malformed_dates})')
        if malformed_ids:
            blockers.append(f'{game}: malformed draw IDs ({malformed_ids})')
        if unordered_ids:
            blockers.append(f'{game}: draw IDs not strictly increasing ({unordered_ids})')
        if missing_sequence_events:
            blockers.append(f'{game}: gaps in consecutive same-year draw IDs ({missing_sequence_events})')
        if invalid_year_rollovers:
            blockers.append(f'{game}: invalid ROC year rollover ({invalid_year_rollovers})')
        if unordered_dates:
            blockers.append(f'{game}: draw dates not strictly increasing ({unordered_dates})')
        if date_year_mismatches:
            blockers.append(f'{game}: ROC year and draw date disagree ({date_year_mismatches})')
        if info.get('first') is not None and str(info['first']) != first_id:
            blockers.append(f'{game}: first draw ID differs from manifest')
        expected_last = info.get('last', info.get('cutoff'))
        if expected_last is not None and str(expected_last) != last_id:
            blockers.append(f'{game}: last draw ID differs from manifest')
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
