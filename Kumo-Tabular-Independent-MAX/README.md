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

## Validated baseline and verification scope (2026-11-18 Asia/Taipei)

The independent GitHub Actions run [#44138889627](https://github.com/aa0908601687-web/Commit-History/actions/runs/44138889627) succeeded at commit `507b945e4257fc4dbf01f794453484943e81a472`.

- `unit`: Python syntax and unit tests passed.
- `install`: local `structured-data-models` SDK installation, `pip check`, and SDK import verification passed.
- `e2e`: actual KumoTabular `small` classification inference completed on the scikit-learn breast-cancer dataset with 300 context and 269 held-out rows; prediction report assertions and artifact upload passed.
- Artifacts: `kumo-install-report` and `kumo-inference-report` in the linked run.

**Scope boundary:** This proves a CPU/GPU runner can install the official local SDK and perform a real model inference smoke test. It does **not** prove a production deployment, a general-purpose import/clean/feature/label pipeline, time-aware walk-forward evaluation, benchmark comparisons, lottery prediction advantage, or a complete 15-stage research product. Those capabilities need separate implementation and test evidence.

To check the **latest** branch state, review the most recent `Kumo Tabular Independent CI` run and inspect the three separate jobs and artifacts. A prior passing run does not certify later commits. Only files under this Kumo directory and its dedicated CI workflow should be modified; OF-LGBM is out of scope.
