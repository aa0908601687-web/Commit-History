"""V6 readiness reference tests; deliberately no claim of improved predictive probability."""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import v6_openfe_feature_research as research

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'WP-OS-Git-MAX'))
from v6_readiness_audit import (audit_csv, prequential, reference_scores,
                                _theory, _hit_distribution, _brier, _pick, main)


def rows(n=520):
    out = []
    for idx in range(n):
        first = sorted({((idx * 7 + j * 5) % 38) + 1 for j in range(6)})
        out.append((str(115000001 + idx), first, idx % 8 + 1))
    return out


class ReadinessAuditTests(unittest.TestCase):
    def test_fixed_features_use_no_draws_before_previous_ten(self):
        sample = rows(40)
        for mode in ('base', 'repeat', 'cooccur', 'all'):
            self.assertEqual(reference_scores(sample, mode),
                             reference_scores(rows(120) + sample[-10:], mode))
            self.assertTrue(all(0 <= p <= 1 for p in reference_scores(sample, mode).values()))
        with self.assertRaises(ValueError):
            reference_scores(sample[:9], 'base')
        with self.assertRaises(ValueError):
            reference_scores(sample, 'unexpected_mode')

    def test_exact_fair_distribution_and_deterministic_selection(self):
        theory = _theory()
        self.assertEqual(len(theory), 7)
        self.assertAlmostEqual(sum(theory.values()), 1.0)
        self.assertEqual(_hit_distribution([0, 2, 2])['2'], 2)
        s = {i: .5 if i <= 6 else .01 for i in range(1, 39)}
        self.assertEqual(_pick(s), set(range(1, 7)))
        self.assertAlmostEqual(_brier(s, set(range(1, 7))),
                               sum((s[i] - int(i <= 6)) ** 2 for i in range(1, 39)) / 38)

    def test_causal_100_200_500_reproducibility_and_no_auto_promotion(self):
        r = rows(515)
        report = prequential(r)
        self.assertEqual(set(report['windows']), {'100', '200', '500'})
        for n in (100, 200, 500):
            window = report['windows'][str(n)]
            self.assertEqual(window['tested_draws'], n)
            self.assertEqual(sum(window['variants']['all']['hit_distribution'].values()), n)
        self.assertFalse(report['openfe_lightgbm_retrained_for_every_holdout'])
        self.assertFalse(report['automatic_production_promotion'])
        self.assertFalse(report['external_model_live_deployment_verified'])
        self.assertEqual(report['history_window_max'], 10)
        self.assertEqual(report, prequential(r))

    def test_last_draw_label_does_not_affect_its_prediction(self):
        r = rows(32)
        altered = list(r)
        altered[-1] = (r[-1][0], [1, 2, 3, 4, 5, 6], 2)
        for mode in ('base', 'all'):
            self.assertEqual(reference_scores(r[:-1], mode), reference_scores(altered[:-1], mode))
        self.assertEqual(prequential(r, lengths=(3,))['windows']['3']['from_draw'],
                         prequential(altered, lengths=(3,))['windows']['3']['from_draw'])

    def test_invalid_holdout(self):
        for lengths in ((), (0,), (500,), (-4,)):
            with self.assertRaises(ValueError):
                prequential(rows(40), lengths=lengths)

    def test_csv_fail_closed_validations(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / 'input.csv'
            def write(data, header=('draw_id', 'first', 'second')):
                with file.open('w', newline='') as f:
                    w = csv.writer(f); w.writerow(header)
                    for row in data:
                        w.writerow(row)
            valid = [(k, ' '.join(map(str, nums)), b) for k, nums, b in rows(515)]
            write(valid)
            parsed, report = audit_csv(file)
            self.assertEqual(len(parsed), 515)
            self.assertEqual(len(report['sha256']), 64)
            self.assertFalse(report['source_verified_independently_against_official_provider'])
            self.assertFalse(report['date_field_present'])
            write(valid[:20])
            with self.assertRaises(ValueError):
                audit_csv(file)
            write(valid, header=('first', 'draw_id', 'second'))
            with self.assertRaises(ValueError):
                audit_csv(file)
            corrupt = [
                valid[:-1] + [valid[0]],
                valid[:-1] + [('bad', '01 02 03 04 05 06', 1)],
                valid[:-1] + [(valid[-1][0], '01 01 03 04 05 06', 1)],
                valid[:-1] + [(valid[-1][0], '01 02 03 04 05 06', 9)],
                valid[:-1] + [(valid[-1][0], 'not a list', 1)],
            ]
            for attempt in corrupt:
                write(attempt)
                with self.assertRaises(ValueError):
                    audit_csv(file)

    def test_cli_emits_complete_artifact(self):
        with TemporaryDirectory() as tmp:
            path, output = Path(tmp) / 'input.csv', Path(tmp) / 'report.json'
            with path.open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['draw_id', 'first', 'second'])
                for key, nums, second in rows(515):
                    w.writerow([key, ' '.join(map(str, nums)), second])
            with patch.object(sys, 'argv', ['audit', '--csv', str(path), '--output', str(output)]):
                main()
            report = json.loads(output.read_text())
            self.assertEqual(report['data_quality']['last_draw'], rows(515)[-1][0])
            self.assertFalse(report['backtest']['openfe_lightgbm_retrained_for_every_holdout'])


    def test_research_training_anchors_are_bounded(self):
        self.assertEqual(len(research._anchors(150, max_anchors=4)), 4)
        with self.assertRaises(ValueError):
            research._anchors(150, max_anchors=1)

    def test_research_rejects_wrong_shape_and_invalid_probability(self):
        with patch('v6_openfe_bridge._fit_and_score', return_value=([.1] * 113, 12, 12)):
            with self.assertRaises(RuntimeError):
                research.score_three_views(rows(15), 'all')
        with patch('v6_openfe_bridge._fit_and_score', return_value=([float('nan')] * 114, 12, 12)):
            with self.assertRaises(RuntimeError):
                research.score_three_views(rows(15), 'all')

    def test_research_default_scorer_calls_the_real_scoring_entrypoint(self):
        scores = {n: .15 for n in range(1, 39)}
        with patch.object(research, 'score_three_views', return_value=scores) as scorer:
            result = research.walk_forward(rows(15), draws=1, modes=('base',))
        scorer.assert_called_once()
        self.assertEqual(result['results']['base']['draws'], 1)

    def test_research_cli_walk_forward_without_output_file(self):
        dummy = SimpleNamespace(validate=lambda _path: rows(15))
        with patch.dict(sys.modules, {'v6_exact_e2e': dummy}):
            with patch.object(research, 'walk_forward', return_value={'research': True}) as run:
                with patch.object(sys, 'argv', ['research', '--csv', 'unused.csv', '--walk-forward-draws', '1',
                                                '--modes', 'base']):
                    research.main()
        run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
