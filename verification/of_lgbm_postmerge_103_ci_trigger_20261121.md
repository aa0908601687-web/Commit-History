# OF-LGBM post-merge validation-only trigger

This file exists only to trigger the existing three-game OpenFE + LightGBM CPU E2E and chronological five-refit holdout workflows following PR #103 squash merge.

Pinned base commit: `0d65387dbe9942c05f9cdd2199bb4b21ee5bca11`.

No algorithms, training logic, prediction selection, historical CSV, or source manifest are changed. This validation-only PR must not be merged.

Acceptance: power/lotto/539 three-game CPU E2E jobs and three five-refit holdout jobs all succeed on this post-merge base. Official provenance, 1,964 missing Power dates, 100-candidate feature integration, and long-horizon predictive advantage remain unverified; no claim of lottery prediction benefit.
