"""Independent tests for opt-in V6 OpenFE feature research; no OpenFE install needed."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'WP-OS-Git-MAX'))
import v6_openfe_feature_research as lab


def sample_rows(count=140):
    rows = []
    for i in range(count):
        numbers = sorted({(i * 7 + j * 5) % 38 + 1 for j in range(6)})
        assert len(numbers) == 6
        rows.append((str(115000001 + i), numbers, i % 8 + 1))
    return rows


class V6OpenFEFeatureResearchTests(unittest.TestCase):
    def test_all_modes_keep_base_data(self):
        rows = sample_rows(20)
        base = lab.features(rows, 1, view_limit=None, mode='base')
        for mode in ('repeat', 'cooccur', 'all'):
            features = lab.features(rows, 1, view_limit=None, mode=mode)
            for name, value in base.items():
                self.assertEqual(features[name], value)
        self.assertNotIn('repeat_last', base)
        self.assertNotIn('pair_peak_5', base)
        self.assertIn('repeat_last', lab.features(rows, 1, view_limit=5, mode='all'))
        self.assertIn('pair_peak_5', lab.features(rows, 1, view_limit=10, mode='all'))

    def test_strict_last_10_feature_limit(self):
        rows = sample_rows(30)
        for mode in lab.MODES:
            for n in (1, 20, 38):
                a = lab.features(rows, n, view_limit=None, mode=mode)
                # Prepend arbitrarily many older draws: all current features must stay invariant.
                b = lab.features(sample_rows(100) + rows, n, view_limit=None, mode=mode)
                self.assertEqual(a, b)

    def test_repeat_features_are_actual_transitions(self):
        rows = sample_rows(10)
        rows[-3] = ('1', [1, 2, 3, 4, 5, 6], 1)
        rows[-2] = ('2', [2, 3, 4, 5, 6, 7], 1)
        rows[-1] = ('3', [1, 2, 3, 4, 5, 6], 1)
        feat = lab._repeat_features(rows, 1)
        self.assertEqual(feat['repeat_after_gap'], 1.0)
        self.assertEqual(feat['repeat_streak_2'], 0.0)
        self.assertEqual(feat['repeat_last'], 1.0)
        self.assertEqual(lab._repeat_features(rows, 2)['repeat_streak_3'], 1.0)

    def test_pair_triple_support_measures_past_only(self):
        rows = [('1', [1, 2, 3, 4, 5, 6], 1)] * 10
        f = lab._cooccur_features(rows, 1)
        self.assertEqual(f['pair_peak_5'], 1.0)
        self.assertEqual(f['triple_peak_10'], 1.0)
        self.assertEqual(lab._cooccur_features(rows, 38)['pair_peak_5'], 0.0)

    def test_training_labels_are_future_only_and_features_are_past_only(self):
        rows = sample_rows(14)
        train, labels = lab.training_records(rows, 'all')
        self.assertEqual(len(train), (14 - 10) * 3 * 38)
        self.assertEqual(len(labels), len(train))
        before = rows[:]
        after = rows[:]
        after[-1] = (after[-1][0], [1, 2, 3, 4, 5, 6], 8)
        earlier_rows, earlier_labels = lab.training_records(before, 'all')
        changed_rows, changed_labels = lab.training_records(after, 'all')
        # The latest draw is a label, not an input feature for the latest training anchor.
        self.assertEqual(earlier_rows, changed_rows)
        self.assertNotEqual(earlier_labels[-38:], changed_labels[-38:])

    def test_invalid_feature_requests(self):
        with self.assertRaises(ValueError):
            lab.features(sample_rows(12), 1, view_limit=5, mode='unknown')
        with self.assertRaises(ValueError):
            lab.features(sample_rows(12), 0, view_limit=5, mode='all')
        with self.assertRaises(ValueError):
            lab._repeat_features(sample_rows(9), 1)
        with self.assertRaises(ValueError):
            lab.training_records(sample_rows(11), 'all')

    def test_walk_forward_calls_only_prior_draws(self):
        rows = sample_rows(20)
        histories = []
        def fake_scorer(known, mode):
            histories.append((known[-1][0], len(known), mode))
            return {n: 0.15 for n in range(1, 39)}
        result = lab.walk_forward(rows, draws=3, modes=('base', 'all'), scorer=fake_scorer)
        self.assertEqual(result['last_evaluated_draw'], rows[-1][0])
        self.assertEqual(histories, [(rows[16][0], 17, 'base'), (rows[16][0], 17, 'all'),
                                     (rows[17][0], 18, 'base'), (rows[17][0], 18, 'all'),
                                     (rows[18][0], 19, 'base'), (rows[18][0], 19, 'all')])
        self.assertFalse(result['results']['all']['eligible_for_manual_review'])
        self.assertFalse(result['results']['all']['activated_in_production'])
        self.assertAlmostEqual(result['fair_draw_uniform_brier'], (6 / 38) * (32 / 38))

    def test_no_auto_promotion_even_when_candidate_wins(self):
        rows = sample_rows(114)
        def fake_scorer(known, mode):
            if mode == 'base':
                return {n: 0.01 for n in range(1, 39)}
            # In tests only: artificial oracle to exercise statistical screening.
            target = set(rows[len(known)][1])
            return {n: 0.99 if n in target else 0.01 for n in range(1, 39)}
        r = lab.walk_forward(rows, draws=100, modes=('base', 'all'), scorer=fake_scorer)
        self.assertTrue(r['results']['all']['eligible_for_manual_review'])
        self.assertFalse(r['results']['all']['activated_in_production'])
        self.assertFalse(r['production_promotion_automatic'])

    def test_invalid_walk_forward(self):
        with self.assertRaises(ValueError):
            lab.walk_forward(sample_rows(15), draws=5)
        with self.assertRaises(ValueError):
            lab.walk_forward(sample_rows(20), draws=2, modes=('all',))
        with self.assertRaises(ValueError):
            lab.walk_forward(sample_rows(20), draws=2, min_review_draws=20)

    def test_scorer_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            lab.walk_forward(sample_rows(20), draws=1, modes=('base',), scorer=lambda _r, _m: {1: .5})

    def test_feature_training_integration_with_stub_backend(self):
        rows = sample_rows(15)
        with patch('v6_openfe_bridge._fit_and_score', return_value=([0.1] * 114, 12, 12)) as fit:
            got = lab.score_three_views(rows, 'all')
        self.assertEqual(len(got), 38)
        self.assertAlmostEqual(got[1], 0.1)
        self.assertEqual(len(fit.call_args[0][2]), 114)
        self.assertIn('pair_peak_5', fit.call_args[0][0].columns)
        self.assertIn('repeat_last', fit.call_args[0][0].columns)


if __name__ == '__main__':
    unittest.main()
