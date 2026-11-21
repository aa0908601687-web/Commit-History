import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('official_gate', Path(__file__).with_name('of_lgbm_official_readiness.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OfficialReadinessTests(unittest.TestCase):
    def test_missing_evidence_is_blocked(self):
        self.assertFalse(module.audit({})['production_official_data_ready'])

    def test_unverified_rows_are_blocked(self):
        m = {'record_by_record_officially_authenticated': False,
             'datasets': {k: {'official_row_verification': 'not_completed', 'sha256': '0' * 64} for k in module.GAMES}}
        self.assertFalse(module.audit(m)['production_official_data_ready'])

    def test_manifest_flags_alone_cannot_certify_official_source(self):
        m = {'record_by_record_officially_authenticated': True,
             'datasets': {k: {'official_row_verification': 'completed', 'sha256': '0' * 64, 'date_status': 'complete'} for k in module.GAMES}}
        r = module.audit(m)
        self.assertFalse(r['production_official_data_ready'])
        self.assertFalse(r['official_authentication_performed_by_this_check'])

    def test_snapshot_hash_rows_and_dates_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'power.csv'
            data.write_text('draw_id,draw_date,first,second\n115000001,2026-01-01,1 2 3 4 5 6,1\n', encoding='utf-8')
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            m = {'datasets': {'power': {'path': 'power.csv', 'rows': 1, 'sha256': digest},
                              'lotto': {}, '539': {}}}
            report = module.audit(m, snapshot_root=root)
            self.assertTrue(report['snapshot_checks']['power']['sha256_matches_manifest'])
            self.assertEqual(report['snapshot_checks']['power']['dated_rows'], 1)
            self.assertFalse(report['production_official_data_ready'])
            data.write_text('draw_id,draw_date,first,second\n115000001,,1 2 3 4 5 6,1\n', encoding='utf-8')
            report = module.audit(m, snapshot_root=root)
            self.assertFalse(report['snapshot_checks']['power']['sha256_matches_manifest'])
            self.assertEqual(report['snapshot_checks']['power']['dated_rows'], 0)
            self.assertTrue(any('SHA-256 mismatch' in b for b in report['blockers']))
            self.assertTrue(any('incomplete draw dates' in b for b in report['blockers']))

    def test_duplicate_id_and_invalid_date_are_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'power.csv'
            data.write_text('draw_id,draw_date,first,second\n115000001,2026-13-01,1 2 3 4 5 6,1\n115000001,2026-01-02,1 2 3 4 5 6,1\n', encoding='utf-8')
            m = {'datasets': {'power': {'path': 'power.csv', 'rows': 2,
                                        'sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                                        'first': '115000001', 'cutoff': '115000002'},
                              'lotto': {}, '539': {}}}
            report = module.audit(m, snapshot_root=root)
            self.assertFalse(report['production_official_data_ready'])
            self.assertEqual(report['snapshot_checks']['power']['invalid_or_duplicate_ids'], 1)
            self.assertEqual(report['snapshot_checks']['power']['malformed_dates'], 1)
            self.assertTrue(any('last draw ID differs' in b for b in report['blockers']))

    def test_reverse_order_and_roc_year_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'power.csv'
            data.write_text('draw_id,draw_date,first,second\n115000002,2026-01-05,1 2 3 4 5 6,1\n115000001,2025-12-30,1 2 3 4 5 6,1\n', encoding='utf-8')
            m = {'datasets': {'power': {'path': 'power.csv', 'rows': 2,
                                        'sha256': hashlib.sha256(data.read_bytes()).hexdigest()},
                              'lotto': {}, '539': {}}}
            report = module.audit(m, snapshot_root=root)
            check = report['snapshot_checks']['power']
            self.assertEqual(check['unordered_ids'], 1)
            self.assertEqual(check['unordered_dates'], 1)
            self.assertEqual(check['date_year_mismatches'], 1)
            self.assertFalse(report['production_official_data_ready'])

    def test_malformed_draw_id(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'power.csv'
            data.write_text('draw_id,draw_date,first,second\nNOTANID,2026-01-01,1 2 3 4 5 6,1\n', encoding='utf-8')
            m = {'datasets': {'power': {'path': 'power.csv', 'rows': 1,
                                        'sha256': hashlib.sha256(data.read_bytes()).hexdigest()},
                              'lotto': {}, '539': {}}}
            report = module.audit(m, snapshot_root=root)
            self.assertEqual(report['snapshot_checks']['power']['malformed_ids'], 1)
            self.assertFalse(report['production_official_data_ready'])

    def test_invalid_winning_numbers_are_blocked(self):
        examples = [
            ('power', 'first,second', '1 1 2 3 4 5,1'),
            ('lotto', 'first,special', '1 2 3 4 5 6,6'),
            ('539', 'first', '1 2 3 4 40'),
        ]
        for game, extra_header, values in examples:
            with self.subTest(game=game), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                data = root / 'draws.csv'
                data.write_text('draw_id,draw_date,' + extra_header + '\n'
                                + '115000001,2026-01-01,' + values + '\n', encoding='utf-8')
                manifest = {'datasets': {game: {
                    'path': 'draws.csv', 'rows': 1,
                    'sha256': hashlib.sha256(data.read_bytes()).hexdigest()
                }}}
                result = module.audit(manifest, snapshot_root=root)
                self.assertEqual(result['snapshot_checks'][game]['invalid_number_rows'], 1)
                self.assertFalse(result['production_official_data_ready'])

    def test_missing_draw_sequence_and_year_rollover(self):
        cases = [
            (['115000001', '115000003'], 'missing_sequence_events'),
            (['115000099', '116000002'], 'invalid_year_rollovers'),
        ]
        for ids, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                data = root / 'power.csv'
                lines = ['draw_id,draw_date,first,second']
                for index, draw_id in enumerate(ids):
                    year = int(draw_id[:3]) + 1911
                    lines.append(f'{draw_id},{year}-01-0{index + 1},1 2 3 4 5 6,1')
                data.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
                manifest = {'datasets': {'power': {
                    'path': 'power.csv', 'rows': 2,
                    'sha256': hashlib.sha256(data.read_bytes()).hexdigest()
                }}}
                result = module.audit(manifest, snapshot_root=root)
                self.assertEqual(result['snapshot_checks']['power'][field], 1)
                self.assertFalse(result['production_official_data_ready'])

    def test_valid_year_rollover_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'power.csv'
            data.write_text(
                'draw_id,draw_date,first,second\\n'
                '115000099,2026-12-31,1 2 3 4 5 6,1\\n'
                '116000001,2027-01-01,1 2 3 4 5 6,1\\n',
                encoding='utf-8')
            manifest = {'datasets': {'power': {
                'path': 'power.csv', 'rows': 2,
                'sha256': hashlib.sha256(data.read_bytes()).hexdigest()
            }}}
            result = module.audit(manifest, snapshot_root=root)
            check = result['snapshot_checks']['power']
            self.assertEqual(check['missing_sequence_events'], 0)
            self.assertEqual(check['invalid_year_rollovers'], 0)

    def test_directory_escape_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            m = {'datasets': {'power': {'path': '../not-in-repo.csv', 'sha256': 'a'*64},
                              'lotto': {}, '539': {}}}
            report = module.audit(m, snapshot_root=d)
            self.assertTrue(any('outside repository' in b for b in report['blockers']))


if __name__ == '__main__':
    unittest.main()
