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
`train_openfe_lightgbm(train_x, train_y, valid_x, valid_y)` fits OpenFE on training rows only and transforms validation data. Caller must implement chronological splitting, leakage-safe historical feature construction, the 100 independent candidate feature sets, 8-number candidate pool, exact combination search, and end-to-end lottery validation. Those components are **not** implemented or connected here.

This is an integration scaffold, not proof of successful GitHub Actions execution, package installation on the user's phone, or improved lottery predictions. Never use future draw outcomes to construct earlier features.
