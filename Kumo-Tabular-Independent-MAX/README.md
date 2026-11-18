# Kumo-Tabular Independent MAX

This is an isolated NVIDIA KumoTabular research installation in the existing Commit-History repository. **It does not alter OF-LGBM.** Its only model SDK is the NVIDIA local `structured-data-models` package, not the hosted NIM client.

## Run locally

Use Python 3.11 on a supported machine:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify_install.py
python -m unittest discover -s tests -v
python smoke_predict.py
```

The first real inference may download model weights. CPU execution is intended for small smoke tests.

## CI

The independent workflow lives at the repository root in `.github/workflows/kumo-tabular-independent.yml`, since GitHub Actions does not discover workflows in nested project folders. It runs only for `kumo-tabular-setup` branch or manual dispatch, and uses `working-directory: Kumo-Tabular-Independent-MAX`.

CI executes syntax/unit tests, dependency install verification and model inference smoke. Installation/import success is not the same as successful end-to-end inference. Inspect each job and artifacts individually.

This is a research engineering tool; no evidence it predicts fair lottery draws better than chance.


## 15 required capabilities — implementation audit

1. Official NVIDIA KumoTabular SDK: requirements.txt and smoke_predict.py (real installation/inference pending CI evidence).
2. Python runtime: GitHub Actions Python 3.11.
3. Dependency management: requirements.txt and pip check; versions are not yet locked.
4. Data ingestion: data_pipeline.validate_csv (CSV); Parquet adapter not yet implemented.
5. Data integrity: CSV schema, target, duplicates and chronological checks.
6. Preprocessing: basic tabular validation only; task-specific encoding remains to be implemented.
7. No-leakage split: data_pipeline.chronological_split (ordering preserved).
8. Model loading: smoke_predict.py (requires model download).
9. Model inference: smoke_predict.py (requires real E2E evidence).
10. Prediction output: prediction_smoke_report.json (per-row export not yet implemented).
11. Accuracy/error metrics: evaluate_classification/evaluate_regression.
12. SHA-256 audit: sha256_file and write_audit (hash is not proof of pre-event timestamp).
13. Unit tests: unittest tests/test_package.py and tests/test_data_pipeline.py.
14. Integration tests: smoke inference integrates SDK and dataset; broader integration pending.
15. E2E and GitHub CI: .github/workflows/kumo-tabular-independent.yml (pending successful workflow run).

**Status:** This is incremental implementation, not a claim of 15/15 deployment or successful end-to-end validation. The upgrade branch stays independent; OF-LGBM files are unchanged. Kumo-Tabular is not demonstrated to predict fair lotteries above chance.
