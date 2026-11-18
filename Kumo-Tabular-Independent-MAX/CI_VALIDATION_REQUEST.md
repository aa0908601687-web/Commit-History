# Kumo-Tabular Independent CI validation

This file records a request to rerun the independent CI pipeline on the `kumo-tabular-setup` branch.

Required checks: `unit`, `install` (including `pip check` and SDK import), and `e2e` (real KumoTabular inference, 269 held-out predictions, report assertions, and artifact upload).

A passing workflow is required before claiming end-to-end success. This branch is isolated from OF-LGBM.

## Revalidation request — 2026-11-18 (Asia/Taipei)

Re-run the isolated Kumo `unit` → `install` → `e2e` pipeline against the current branch, including genuine KumoTabular inference and report assertions. This commit requests a fresh CI run; the outcome must be checked from the workflow jobs and artifacts. OF-LGBM remains out of scope.
