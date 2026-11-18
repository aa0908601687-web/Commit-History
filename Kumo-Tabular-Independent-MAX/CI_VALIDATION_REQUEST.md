# Kumo-Tabular Independent CI validation

This file records a request to rerun the independent CI pipeline on the `kumo-tabular-setup` branch.

Required checks: `unit`, `install` (including `pip check` and SDK import), and `e2e` (real KumoTabular inference, 269 held-out predictions, report assertions, and artifact upload).

A passing workflow is required before claiming end-to-end success. This branch is isolated from OF-LGBM.

## Revalidation request — 2026-11-18 (Asia/Taipei)

Re-run the isolated Kumo `unit` → `install` → `e2e` pipeline against the current branch, including genuine KumoTabular inference and report assertions. This commit requests a fresh CI run; the outcome must be checked from the workflow jobs and artifacts. OF-LGBM remains out of scope.

## Independent install and E2E revalidation — request 2026-11-18 B

Run dedicated Kumo tests: unit → install (structured-data-models and dependency verification) → e2e (real KumoTabular classification inference and JSON assertions) → scope-audit (15-stage capabilities and gaps). Verify results using this commit's GitHub Actions jobs and uploaded artifacts. This request does not change OF-LGBM or claim 15-stage completion.

## Fresh validation request — 2026-11-18 C

Re-run the independent `unit`, `install`, `e2e`, and `scope-audit` jobs at this commit. The E2E job must execute real KumoTabular inference and the chronological synthetic-data research pipeline, including walk-forward and baseline comparison. Record the final run ID, conclusions and report artifacts; no claim of real-world prediction accuracy. OF-LGBM is excluded.
