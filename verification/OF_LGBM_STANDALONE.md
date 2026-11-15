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
