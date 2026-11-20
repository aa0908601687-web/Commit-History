# OF-LGBM validation request — 2026-11-19

This file is a no-code-change validation trigger for the existing GitHub Actions pull_request workflows:
- `.github/workflows/of-lgbm-23-operator-e2e.yml` (power, lotto, 539)
- `.github/workflows/of-lgbm-forward-audit.yml` (power, lotto, 539; five causal ML refits each)

Baseline main commit: `cad56312f360c91b0dd0900306c504b85188f263`.

Validation must preserve known caveats:
- Official historical CSV rows have not been independently verified.
- Power history has 1,964 missing draw dates.
- Brier evaluation is not selection or probability recalibration.
- Passing CI does not establish predictive advantage or production readiness.

No code, model weights, historical data, dependencies, or main branch files are changed by this verification request. Do not merge this draft validation-only PR.
