#!/usr/bin/env python3
"""Opt-in V6 OpenFE research: repeat/co-occurrence features and honest prequential evaluation.

Research-only. No changes to the production selector, feature weights or deployment.
Every predictive feature uses at most the 10 completed draws preceding the target.
Longer history is used ONLY for earlier, causal training anchors / evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / 'WP-OS-Git-MAX'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

MODES = ('base', 'repeat', 'cooccur', 'all')
VIEWS = {'latest_5': 5, 'latest_10': 10, 'full_archive': None}


def _draws(history: list, limit: int) -> list[set[int]]:
    """Only inspect the last <=10 draws; earlier training anchors are never features."""
    if len(history) < 10:
        raise ValueError('at least 10 completed draws required')
    return [set(row[1]) for row in history[-limit:]]


def _repeat_features(history: list, number: int) -> dict[str, float]:
    d10 = _draws(history, 10)
    d5 = d10[-5:]
    return {
        'repeat_last': float(number in d10[-1]),
        'repeat_streak_2': float(all(number in d for d in d10[-2:])),
        'repeat_streak_3': float(all(number in d for d in d10[-3:])),
        'repeat_transitions_5': float(sum(number in d5[i - 1] and number in d5[i] for i in range(1, 5))),
        'repeat_transitions_10': float(sum(number in d10[i - 1] and number in d10[i] for i in range(1, 10))),
        'repeat_after_gap': float(number in d10[-1] and number not in d10[-2] and number in d10[-3]),
    }


def _max_pair_triple(draws: list[set[int]], number: int) -> tuple[float, float]:
    pairs: Counter[int] = Counter()
    triples: Counter[tuple[int, int]] = Counter()
    for draw in draws:
        if number not in draw:
            continue
        companions = sorted(draw - {number})
        pairs.update(companions)
        triples.update(combinations(companions, 2))
    return float(max(pairs.values(), default=0)), float(max(triples.values(), default=0))


def _cooccur_features(history: list, number: int) -> dict[str, float]:
    d10 = _draws(history, 10)
    pair5, triple5 = _max_pair_triple(d10[-5:], number)
    pair10, triple10 = _max_pair_triple(d10, number)
    # Rates allow the same model to distinguish five-draw and ten-draw supports.
    return {
        'pair_peak_5': pair5 / 5,
        'pair_peak_10': pair10 / 10,
        'triple_peak_5': triple5 / 5,
        'triple_peak_10': triple10 / 10,
    }


def features(history: list, number: int, *, view_limit: int | None, mode: str) -> dict[str, float]:
    if mode not in MODES:
        raise ValueError('invalid feature mode')
    if not 1 <= number <= 38:
        raise ValueError('first zone number out of range')
    from v6_openfe_bridge import _base_row
    result = _base_row(history, number, view_limit)
    if mode in ('repeat', 'all'):
        result.update(_repeat_features(history, number))
    if mode in ('cooccur', 'all'):
        result.update(_cooccur_features(history, number))
    return result


def _anchors(length: int, max_anchors: int) -> list[int]:
    # Matches current production's <=80 uniformly spaced training anchors.
    ends = list(range(10, length))
    if len(ends) > max_anchors:
        if max_anchors < 2:
            raise ValueError('max_anchors must be at least 2')
        positions = {round(i * (len(ends) - 1) / (max_anchors - 1)) for i in range(max_anchors)}
        ends = [ends[i] for i in sorted(positions)]
    return ends


def training_records(rows: list, mode: str, *, max_anchors: int = 80) -> tuple[list[dict], list[int]]:
    """At anchor t: feature history[:t], label rows[t], never rows[t:] as features."""
    if len(rows) < 12:
        raise ValueError('at least 12 known draws required')
    samples: list[dict] = []
    labels: list[int] = []
    for end in _anchors(len(rows), max_anchors):
        past = rows[:end]
        actual = set(rows[end][1])
        for view_limit in VIEWS.values():
            for number in range(1, 39):
                samples.append(features(past, number, view_limit=view_limit, mode=mode))
                labels.append(int(number in actual))
    return samples, labels


def score_three_views(rows: list, mode: str, *, max_anchors: int = 80, max_features: int = 12) -> dict[int, float]:
    """Use the SAME OpenFE + LightGBM fit function as production, research-only."""
    import pandas as pd
    from v6_openfe_bridge import _fit_and_score
    train, labels = training_records(rows, mode, max_anchors=max_anchors)
    current = [features(rows, n, view_limit=v, mode=mode) for v in VIEWS.values() for n in range(1, 39)]
    probabilities, _, _ = _fit_and_score(pd.DataFrame(train), pd.Series(labels, name='selected'),
                                        pd.DataFrame(current), n_jobs=1, max_features=max_features)
    if len(probabilities) != 38 * len(VIEWS):
        raise RuntimeError('OpenFE prediction/view alignment mismatch')
    score = {n: float(sum(probabilities[view * 38 + n - 1] for view in range(3)) / 3)
             for n in range(1, 39)}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in score.values()):
        raise RuntimeError('invalid OpenFE probability')
    return score


def _sign_test(values: list[float]) -> dict[str, float | int]:
    wins = sum(v > 1e-12 for v in values)
    losses = sum(v < -1e-12 for v in values)
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n) if n else 1.0
    return {'wins': wins, 'losses': losses, 'ties': len(values) - n, 'one_sided_p': p}


def _result_snapshot(mode: str, briers: list[float], hits: list[int], diffs: list[float] | None,
                     *, min_review_draws: int, comparisons: int) -> dict:
    result = {
        'mode': mode,
        'draws': len(briers),
        'mean_brier': sum(briers) / len(briers),
        'hits_3plus': sum(h >= 3 for h in hits),
        'hit_distribution': {str(n): hits.count(n) for n in range(7)},
    }
    if diffs is not None:
        split = len(diffs) // 2
        improved_halves = split > 0 and sum(diffs[:split]) > 0 and sum(diffs[split:]) > 0
        sign = _sign_test(diffs)
        result.update({
            'paired_brier_gain_vs_base': sum(diffs) / len(diffs),
            'paired_sign_test': sign,
            'eligible_for_manual_review': bool(
                len(diffs) >= min_review_draws
                and sum(diffs) > 0
                and improved_halves
                and sign['one_sided_p'] < .05 / max(1, comparisons)
            ),
            'activated_in_production': False,
        })
    return result


def walk_forward(rows: list, *, draws: int = 20, modes: tuple[str, ...] = MODES,
                 scorer=None, min_review_draws: int = 100) -> dict:
    """Each selection trains strictly before its held-out target draw.

    Early results are smoke tests, NEVER proof of predictive improvement.
    The null baseline is the calibrated fair-draw marginal of 6/38 per number.
    """
    if draws < 1 or len(rows) - draws < 12:
        raise ValueError('evaluation needs >=12 earlier draws and >=1 test draw')
    if not modes or 'base' not in modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError('modes must be unique and include base')
    if min_review_draws < 100:
        raise ValueError('minimum review sample cannot be below 100 draws')
    if scorer is None:
        scorer = score_three_views
    losses = {m: [] for m in modes}
    hits = {m: [] for m in modes}
    uniform = []
    for target_index in range(len(rows) - draws, len(rows)):
        known = rows[:target_index]
        target = set(rows[target_index][1])
        uniform.append(sum((6 / 38 - int(n in target)) ** 2 for n in range(1, 39)) / 38)
        for mode in modes:
            scores = scorer(known, mode)
            if set(scores) != set(range(1, 39)) or any(not math.isfinite(scores[n]) or not 0 <= scores[n] <= 1 for n in scores):
                raise ValueError('scorer must give probabilities for every number')
            picked = sorted(scores, key=lambda n: (-scores[n], n))[:6]
            hits[mode].append(len(set(picked) & target))
            losses[mode].append(sum((scores[n] - int(n in target)) ** 2 for n in range(1, 39)) / 38)
    results = {}
    for mode in modes:
        differences = None if mode == 'base' else [a - b for a, b in zip(losses['base'], losses[mode])]
        results[mode] = _result_snapshot(mode, losses[mode], hits[mode], differences,
                                         min_review_draws=min_review_draws, comparisons=len(modes) - 1)
    return {
        'evaluation': 'prequential: fit before target, then compare with revealed target',
        'last_evaluated_draw': rows[-1][0],
        'first_evaluated_draw': rows[-draws][0],
        'training_features_max_completed_draws': 10,
        'fair_draw_uniform_brier': sum(uniform) / len(uniform),
        'feature_selection': 'holdout paired Brier + Bonferroni-corrected sign test + two-half stability',
        'production_selector_unchanged': True,
        'production_promotion_automatic': False,
        'results': results,
        'caveat': 'research metrics only; not evidence of better lottery odds without independent prospective tests',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--walk-forward-draws', type=int, default=0)
    ap.add_argument('--modes', default='base,repeat,cooccur,all')
    ap.add_argument('--output')
    args = ap.parse_args()
    from v6_exact_e2e import validate
    rows = validate(args.csv)
    if args.walk_forward_draws:
        modes = tuple(m.strip() for m in args.modes.split(','))
        result = walk_forward(rows, draws=args.walk_forward_draws, modes=modes)
    else:
        result = {
            'mode': 'features_only', 'last_known_draw': rows[-1][0],
            'feature_window': [5, 10], 'production_selector_unchanged': True,
            'example': features(rows, 1, view_limit=None, mode='all'),
            'warning': 'no model trained and no walk-forward evaluation was run',
        }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + '
'
    if args.output:
        Path(args.output).write_text(serialized, encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
