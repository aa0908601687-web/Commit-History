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
