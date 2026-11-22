# OpenFE + LightGBM optional integration

This directory adds an isolated research integration without changing the existing lottery workflows.

## Install
Use Python 3.10+ in an isolated environment:
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r of_lgbm/requirements.txt
```
OpenFE upstream recommends `pip install openfe`, not `conda install openfe`.
Upstream source: https://github.com/IIIS-Li-Group/OpenFE

## Test
```bash
python -m unittest discover -s of_lgbm -p 'test_*.py'
```

## Usage and limitations
`train_openfe_lightgbm(train_x, train_y, valid_x, valid_y)` fits OpenFE on training rows only and transforms validation data. Caller must implement chronological splitting, leakage-safe historical feature construction, 8-number candidate scoring, and end-to-end lottery validation. Those components are **not** implemented or connected here.\n\nAn optional standalone `exact_search.select_six_from_eight(scores)` helper now exhaustively checks all 28 six-number combinations from exactly eight first-area candidates (01–38) using an **additive score objective**. It is not wired into the existing V6 workflow, does not model second-area numbers, and must not be represented as the complete V6 exact-search objective.

This is an integration scaffold, not proof of successful GitHub Actions execution, package installation on the user's phone, or improved lottery predictions. Never use future draw outcomes to construct earlier features.

## Isolated selection integration
`integration.select_from_model_probabilities(first_numbers, first_probabilities, second_numbers, second_probabilities)` now connects **precomputed** first-area (1–38) and second-area (1–8) model scores to the eight-candidate pool, additive exact six-number search, and second-area top-one selection. The existing `of_lgbm/test_integration.py` exercises this contract in CI using deterministic synthetic scores. It does **not** fit OpenFE or LightGBM in that test, load verified draw CSVs, build or implement chronological walk-forward training. Existing V6 readonly E2E runs are distinct and must not be treated as proof this adapter is connected to the production pipeline.

## Model-to-selection bridge
`model_bridge.train_and_select(...)` now calls the isolated `train_openfe_lightgbm` function separately for first and second areas, extracts the positive-class probability from each model, and passes all 38/8 scores into the deterministic 8-to-6 + second-area selector. The new `test_model_bridge.py` verifies the handoff with a fake trainer, without requiring historical CSV or model fitting. For future inference, `first_valid_y` and `second_valid_y` are length-matching placeholders required by the current trainer interface; **never supply unknown future outcomes or use them to construct features**. This is an adapter only: verified chronological CSV ingestion, proper pre-draw feature generation, real-data training, and independent out-of-sample testing remain unfinished.

## Historical CSV format adapter
`history_adapter.load_verified_format(path)` validates an explicitly normalized local CSV with columns `draw_id,date,n1,n2,n3,n4,n5,n6,second`, ISO dates, first-area 1–38, second-area 1–8, and strict chronological ordering. `historical_windows(rows, target_index)` returns last-10, last-20, and all-prior views without including the target draw. This validates **format only**, not official-source authenticity. A separate research runner now links this adapter to pre-draw features and model training; it is not a production ingestion pipeline. It does not assume raw Taiwanese lottery CSV exports already match this schema.

## Three-view pre-draw feature builder
`feature_builder.build_number_features(rows, target_index, area)` builds per-number counts and rates from the last 10, last 20, and all earlier draws, never the target draw. `build_supervised_history(rows, target_indices, area)` builds labeled historical training examples using only past draw labels. This is a minimal deterministic baseline; OpenFE's full automatic feature generation, the actual training and prediction on verified lottery draws, and CI end-to-end data/model validation remain to be completed. No fixed cross-view weights are imposed.

## Opt-in CSV-to-selection research runner
`research_runner.run_research_csv(path)` now connects normalized historical CSV validation, chronological three-view per-number features, separate first/second-area OpenFE+LightGBM training, eight-candidate exact additive selection, and SHA-256 of the input CSV. It is **not** wired to production V6 or an official live feed. `test_research_runner.py` checks this path with a fake model; actual OpenFE+LightGBM training on verified draws, blind walk-forward evaluation, and real-data E2E certification remain unverified. The runner uses known historical labels only for training; next-draw labels are never read.

## Current research scope (November 2026)
The configured V6 OF3Q architecture uses 10-draw, 20-draw and all-prior historical views, a single LightGBM model, OpenFE feature engineering, an eight-number first-area candidate pool, additive exact six-number selection, and SHA-256 integrity checks. The former 100 independent candidate-feature configurations are not a current requirement. This directory is an isolated research implementation; the GitHub main-branch historical snapshots are not authenticated complete official exports. Passing CI verifies the tested contracts, not source provenance, prospective accuracy or production deployment.
