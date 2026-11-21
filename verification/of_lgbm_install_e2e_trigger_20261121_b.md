# OF LGBM independent install and E2E verification

Validation-only trigger against current `main`, 2026-11-21.

This file triggers existing GitHub Actions pull-request workflows without changing the model, requirements, data CSV, manifest, or selection behavior.

Acceptance criteria:
- Each of three Python 3.11 CPU jobs installs dedicated OpenFE 0.0.12 and LightGBM dependencies.
- Each of Power, Lotto, and Daily539 completes all-23-operator end-to-end execution.
- Each of the three games completes its five chronological LightGBM refits and holdout gates.
- Preserve the official-source provenance risk and 1,964 missing Power dates; passing E2E is not official CSV installation or predictive superiority.

This PR is testing-only. DO NOT MERGE into main.
