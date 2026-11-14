# V6 OF3Q experimental TabPrep feature engineering

Upstream: https://github.com/atschalz/tabprep (Apache-2.0). This project is an **optional research alternative** to the current OpenFE + LightGBM path. It is not enabled by default and is not evidence of a lottery prediction advantage.

## Installation
Use Python 3.11 and consult the **current upstream README** before installing. TabPrep's sklearn wrapper is in the upstream `tabprep.py` and depends on the AutoGluon source feature generators. Upstream currently documents cloning `autogluon`, running `full_install.sh`, cloning `tabarena`, and installing its benchmark dependencies. Ensure `from tabprep import TabPrepFeatureGenerator` works in the V6 runner's Python environment. Do not blindly install an unrelated PyPI package named `tabprep`.

Also install `lightgbm`, `pandas`, and `scikit-learn`. Upstream installations are substantial; keep them separate from the existing production CPU workflow until measured.

## Usage
```bash
python WP-OS-Git-MAX/v6_exact_e2e.py \
  --csv WP-OS-Git-MAX/historical_archive.csv \
  --tabprep-three-views \
  --output v6-tabprep-experiment.json
```

This is an **isolated alternative**: `--tabprep-three-views` cannot combine with `--openfe-three-views`, `--openfe`  in the first experiment. The OpenFE/LightGBM pipeline remains the primary scoring path. This workflow's historical filename remains for backward compatibility, and is **not** proof that non-variational CUDA-Q is removed.

## Validation and safety
- Source generation and `fit_transform` use only completed historical draws before each training target, with only the preceding 5/10 draws used for feature calculations.
- `transform` applies the fitted transform to current candidates without future labels.
- The initial implementation explicitly disables target encoding (`use_oofte=False`) to reduce leakage risk.
- The three views are 5 draws, 10 draws, and full-archive **training anchor** view (still features capped at 10).
- Run `python -m unittest WP-OS-Git-MAX/test_v6_tabprep_bridge.py -v` for offline contract tests.
- Before enabling the method in the main route, install the upstream dependencies, perform a real TabPrep run, compare against unchanged OpenFE in pre-registered chronological walk-forward tests, and reject if there is no repeatable out-of-sample benefit.
- Passing offline contract tests does **not** prove TabPrep is installed or executable in GitHub Actions. No GPU, NVIDIA-native inference, or real QPU is required.
