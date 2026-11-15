# OF-LGBM Standalone: OpenFE + LightGBM CPU E2E

This is the **independent OF-LGBM research workflow**, not the repository's legacy integrated lottery, V6, or quantum test harness.

## Components in this workflow

- `verification/of_lgbm_all_operator_e2e.py`: 23 OpenFE feature operator families, LightGBM classification, three-view score fusion, and exact combination enumeration in Python.
- `verification/test_of_lgbm_all_operator_e2e.py`: history-window/feature/operator tests.
- `verification/test_of_lgbm_isolation.py`: dependency and import isolation contracts.
- `verification/of_lgbm_requirements.txt`: dedicated, minimal CPU dependencies.
- `verification/of_lgbm_data/power_history_archive.csv`: repository copy of an existing historical Power CSV used only as an offline E2E fixture.
- `.github/workflows/of-lgbm-23-operator-e2e.yml`: dedicated CI job matrix for Power, Lotto, and 539.

**The standalone runner does not import or execute CUDA-Q, dimod/Ising/QUBO, QAOA, or V6 code.** Its final selector uses Python `itertools.combinations`. Other workflows elsewhere in the shared repository can still be reported by GitHub on the *same PR*; they are not part of the OF-LGBM E2E and have not been deleted.

## Reproduce on Python 3.11

```bash
python -m pip install -r verification/of_lgbm_requirements.txt
python -m unittest verification/test_of_lgbm_isolation.py -v
python -m unittest verification/test_of_lgbm_all_operator_e2e.py -v

# Actual historical CSV input, not an independent official-source certification
python verification/of_lgbm_all_operator_e2e.py --game power \
  --csv verification/of_lgbm_data/power_history_archive.csv \
  --output power-of-lgbm-evidence.json

# Deterministic SYNTHETIC fixtures for these two games, not real prediction data
python verification/of_lgbm_all_operator_e2e.py --game lotto --output lotto-synthetic.json
python verification/of_lgbm_all_operator_e2e.py --game 539 --output 539-synthetic.json
```

For fully historical Lotto and 539 E2E, supply **their own** chronological, source-verified CSV files with `--csv`. Do not describe synthetic model runs as real-data predictions.

## Data and scoring boundaries

The three independently trained perspectives have weights 35% recent-5, 35% recent-10, 30% archive training anchors. The archive view samples up to 100 anchors spread through history, but each anchor's feature vector only uses the previous 5 or 10 **completed** draws. The historical Power CSV in this directory was copied from the existing repository file `WP-OS-Git-MAX/historical_archive.csv` on 2026-11-15 to avoid runtime cross-project file dependencies; its authenticity and complete official provenance were not independently verified by that copy.

The Power second zone uses a simple 5/10 frequency baseline, **not** a trained LightGBM model. The standalone E2E is an implementation and reproducibility test, **not** proof of improved lottery odds, official-current-data verification, or a production forecasting deployment.


## Historical integrity audit and prequential ML check

The standalone historical audit is completely separate from the legacy integrated runner. The new `verification/of_lgbm_forward_audit.py` checks chronological identifiers (including year rollover), duplicates, number range, historical row count and SHA-256. It does NOT authenticate all rows against the official source or establish lottery prediction advantage.

```bash
python -m unittest verification/test_of_lgbm_forward_audit.py -v
python verification/of_lgbm_forward_audit.py --game power --csv verification/of_lgbm_data/power_history_archive.csv --folds 0 --output integrity.json
python verification/of_lgbm_forward_audit.py --game power --csv verification/of_lgbm_data/power_history_archive.csv --folds 1 --output forward.json
# Longer runs are available but expensive; do not label them completed without execution:
# --folds 100, --folds 200, --folds 500
```

The `--folds 1` validation fits the actual OpenFE 23-operator plus LightGBM model separately for the single target's earlier data; this is a causal one-step smoke test, NOT a 100/200/500-draw full machine-learning walk-forward. The new `.github/workflows/of-lgbm-forward-audit.yml` runs these checks on Python 3.11.

A limited official-source SPOT CHECK was manually recorded on 2026-11-15: Power draw `115000091`, dated 2026-11-12, first-zone `07 09 15 23 25 35`, second-zone `02`. This agrees with the final archived CSV line and the public lottery result page at https://lotto.ctbcbank.com/result_all.htm . This checks ONE record; the other historical rows were not independently matched to the official publisher. The archive has no date column, and source provenance is not established by SHA-256 alone.

The PR removes its changes to the old independent-power-history integration script and associated CUDA-Q/QUBO workflow; the standalone runner does not import the legacy modules. The existing `main` code is left unchanged. Since `main` has diverged from the PR branch, evaluate the combined merge diff and current CI before promoting the draft.
