# V6 OF3Q OpenFE candidate features: independent research

This is **opt-in, offline and research-only**. It does not change the production V6 OF3Q selector, current `main` model or output, default weights, CUDA-Q CPU backend, Ising/QUBO, Rocq, or second-zone selection. QAOA, PyTorch, ABE and physical QPU are **not** enabled.

## Features

`verification/v6_openfe_feature_research.py` extends the existing `v6_openfe_bridge._base_row` with two separate candidate blocks:

- **Repeat/streak**: previous-draw reuse, two-/three-draw streaks, repeated transitions in last five/ten completed draws, return after a one-draw gap.
- **Co-occurrence**: highest two-number and three-number support involving each candidate within the last five and ten completed draws. Support is a descriptive past statistic, **not** a future winning probability. The short windows and highly sparse combinations can overfit.

Baseline `base`, `repeat`, `cooccur`, and `all` options are exposed. The feature history for any training or predicted target never exceeds the prior ten completed draws. Historical data older than ten draws is permitted only to construct causal training *examples* and held-out evaluations. The `full_archive` view is a ten-draw decayed feature view, with history serving as training anchors; it is **not** a longer selection feature window.

## Scoring and independent screening

The research scorer calls the repository's existing OpenFE and LightGBM `_fit_and_score` implementation; per-number probabilities from three views are equally averaged. This code does **not** replace the production function. Every walk-forward prediction is trained on records preceding the test draw, and the test label is compared only after scoring. It calculates marginal Brier scores across 38 first-zone numbers, the 0-6 hits distribution, 3+ hits and a fair baseline Brier score using probability 6/38 per number.

Feature screening uses **paired holdout Brier gain over the unchanged baseline**, an exact one-sided paired sign test with Bonferroni correction across candidate blocks, and positive gains in both halves. At least 100 held-out draws are required before a candidate can even be **flagged for manual review**. These diagnostics are preliminary and are not a replacement for untouched prospective validation, proper model selection isolation, or a matched full-search null. **No automatic feature promotion exists**. A winning candidate still requires independent future-draw validation and human code review. In fair lotteries no feature has been demonstrated to improve single-ticket odds.

## Run

```bash
# Feature-only preview: no OpenFE fit or backtest
python verification/v6_openfe_feature_research.py \
  --csv WP-OS-Git-MAX/historical_archive.csv

# Actual OpenFE+LightGBM candidate research, can take substantial CPU time
python verification/v6_openfe_feature_research.py \
  --csv WP-OS-Git-MAX/historical_archive.csv \
  --walk-forward-draws 100 --modes base,repeat,cooccur,all \
  --output /tmp/v6-openfe-feature-evidence.json
```

Install OpenFE v0.0.12 and compatible dependencies (as in `.github/workflows/v6-openfe-integration.yml`), or run the included GitHub Actions workflow. The workflow runs unit tests and **one-draw smoke test only**: a green workflow does not establish statistical superiority. Tests use mocks for boundary checks; the workflow smoke test uses the actual OpenFE+LightGBM scorer. Production weights, `main` selection, and live data permissions are untouched.
