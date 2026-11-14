#!/usr/bin/env python3
"""Deterministic V6 audit: data checks and causal 100/200/500-draw reference baselines.

Not full OpenFE retraining, a GPU benchmark, or evidence of better lottery odds.
Feature windows are strictly the ten completed draws before each target.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from v6_openfe_feature_research import features

FIRST_SIZE = 38
PICK = 6
FAIR_MARGINAL = PICK / FIRST_SIZE
WINDOWS = (100, 200, 500)
MODES = ('base', 'repeat', 'cooccur', 'all')


def audit_csv(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ['draw_id', 'first', 'second']:
            raise ValueError('unexpected historical CSV header')
        data = list(reader)
    if len(data) < max(WINDOWS) + 12:
        raise ValueError('archive too short for 500-draw causal evaluation')
    parsed, used = [], set()
    for row in data:
        draw_id = row['draw_id']
        try:
            first = [int(x) for x in row['first'].split()]
            second = int(row['second'])
        except (ValueError, AttributeError) as exc:
            raise ValueError('invalid historical number encoding') from exc
        if (not isinstance(draw_id, str) or len(draw_id) != 9 or not draw_id.isdigit()
                or draw_id in used or (parsed and int(draw_id) <= int(parsed[-1][0]))
                or len(first) != PICK or len(set(first)) != PICK
                or any(not 1 <= n <= FIRST_SIZE for n in first)
                or not 1 <= second <= 8):
            raise ValueError('invalid/duplicate/unsorted draw or number: ' + str(draw_id))
        used.add(draw_id)
        parsed.append((draw_id, first, second))
    return parsed, {
        'row_count': len(parsed), 'first_draw': parsed[0][0], 'last_draw': parsed[-1][0],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'csv_validated': True, 'source_verified_independently_against_official_provider': False,
        'date_field_present': False, 'date_and_draw_calendar_completeness_verified': False,
        'per_decision_feature_windows': [5, 10],
    }


def reference_scores(history, mode):
    """Fixed, untrained CPU proxy sharing the research project's feature definitions."""
    if mode not in MODES or len(history) < 10:
        raise ValueError('invalid mode or insufficient prior draws')
    out = {}
    for n in range(1, FIRST_SIZE + 1):
        f = features(history, n, view_limit=None, mode=mode)
        p = .30 * FAIR_MARGINAL + .45 * (f['freq_5'] / 5) + .25 * (f['freq_10'] / 10)
        if mode in ('repeat', 'all'):
            p += .008 * f['repeat_last'] + .004 * f['repeat_streak_2']
        if mode in ('cooccur', 'all'):
            p += .006 * f['pair_peak_5'] + .003 * f['triple_peak_10']
        out[n] = min(1.0, max(0.0, p))
    return out


def _brier(scores, actual):
    return sum((scores[n] - int(n in actual)) ** 2 for n in range(1, FIRST_SIZE + 1)) / FIRST_SIZE


def _pick(scores):
    return set(sorted(scores, key=lambda n: (-scores[n], n))[:PICK])


def _hit_distribution(hits):
    return {str(i): hits.count(i) for i in range(PICK + 1)}


def _theory():
    total = math.comb(FIRST_SIZE, PICK)
    return {str(k): math.comb(PICK, k) * math.comb(FIRST_SIZE - PICK, PICK - k) / total
            for k in range(PICK + 1)}


def prequential(rows, *, lengths=WINDOWS, seed=20261114):
    """Historical test draws never participate in that draw's feature computation."""
    if not lengths or any(not isinstance(n, int) or n < 1 or len(rows) - n < 10 for n in lengths):
        raise ValueError('invalid holdout lengths')
    end, longest = len(rows), max(lengths)
    records = []
    for target_idx in range(end - longest, end):
        history = rows[:target_idx]
        actual = set(rows[target_idx][1])
        rand = set(random.Random(seed + target_idx).sample(range(1, FIRST_SIZE + 1), PICK))
        hits = {
            'uniform_random': len(rand & actual),
            'fixed_1_to_6': len(set(range(1, PICK + 1)) & actual),
        }
        uniform = _brier({n: FAIR_MARGINAL for n in range(1, FIRST_SIZE + 1)}, actual)
        briers = {'uniform_probability': uniform}
        for mode in MODES:
            scores = reference_scores(history, mode)
            hits[mode] = len(_pick(scores) & actual)
            briers[mode] = _brier(scores, actual)
        records.append({'target_draw': rows[target_idx][0], 'hits': hits, 'brier': briers})
    theory = _theory()
    batches = {}
    for length in lengths:
        subset = records[-length:]
        variants = {}
        for name in subset[0]['hits']:
            hits = [x['hits'][name] for x in subset]
            variants[name] = {
                'hit_distribution': _hit_distribution(hits),
                'hits_3plus': sum(h >= 3 for h in hits),
                'hits_3plus_percent': 100 * sum(h >= 3 for h in hits) / length,
            }
            if name in MODES:
                variants[name]['mean_brier'] = sum(x['brier'][name] for x in subset) / length
        batches[str(length)] = {
            'tested_draws': length,
            'from_draw': subset[0]['target_draw'],
            'through_draw': subset[-1]['target_draw'],
            'fair_theoretical_expected_3plus_draws': length * sum(theory[str(k)] for k in range(3, PICK + 1)),
            'fair_theoretical_marginal_brier': FAIR_MARGINAL * (1 - FAIR_MARGINAL),
            'fair_empirical_marginal_brier': sum(x['brier']['uniform_probability'] for x in subset) / length,
            'variants': variants,
        }
    return {
        'engine': 'CPU fixed-reference feature-only audit (NOT OpenFE retraining)',
        'openfe_lightgbm_retrained_for_every_holdout': False,
        'feature_selection_on_holdout': False,
        'automatic_production_promotion': False,
        'history_window_max': 10,
        'random_seed': seed,
        'fair_exact_first_zone_hit_probabilities': theory,
        'external_model_live_deployment_verified': False,
        'caveat': 'No evidence of better lottery odds; reference audit is not full OpenFE walk-forward.',
        'windows': batches,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--output', default='v6-readiness-audit.json')
    args = ap.parse_args()
    rows, quality = audit_csv(args.csv)
    result = {'data_quality': quality, 'backtest': prequential(rows)}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'data_quality': quality, 'windows': {k: v['tested_draws'] for k, v in result['backtest']['windows'].items()},
                      'actual_openfe_retrained': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
