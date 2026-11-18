# OF-LGBM three-view scoring update — implementation contract (2026-11-19)

Status: **design-only draft; not implemented or E2E validated**.

## Requested behavior
- Use only target-prior historical draws. Build 10-draw, 20-draw, and all-prior-history feature perspectives for OpenFE and LightGBM.
- Learn the influence of these perspectives **from training data only** using LightGBM (e.g. one pooled feature matrix with view-prefixed features). Do not claim that averaging three independently trained LightGBM models is automatically learned fusion.
- Remove `causal_view_weights` / historical-frequency Brier-loss based weights from ticket-selection scoring in `verification/of_lgbm_all_operator_e2e.py`.
- Keep Brier score **evaluation only**, computed on chronological out-of-sample predictions, with no feedback into current ticket scores or view weights.
- Preserve 23 OpenFE operator families, all three lottery games, 8-number pool and exact candidate search, data provenance checks, SHA-256 and strict no-future-data cutoff.
- Preserve the existing separate 5% pair co-occurrence selection term pending explicit owner decision; do not silently remove it.

## Current implementation discrepancy
As checked on main on 2026-11-19, `run()` invokes `causal_view_weights(draws, maximum)` and computes `fused = sum(view_weights[k] * results[k] for k in view_weights)`. The view weights come from historical-frequency Brier losses rather than LightGBM-learned feature importances. The pair-selection module has `PAIR_WEIGHT = 0.05`.

## Acceptance criteria before merge
1. Train/predict with strictly prior-only 10/20/full views and one LightGBM model; no Brier-derived selection weights.
2. Verify per-view feature availability and no target/future leakage with adversarial tests.
3. Brier reported only on forward predictions; changing Brier evaluation results must not change selected tickets.
4. Existing pair 5% remains unchanged unless separately authorized.
5. Run all three games' standalone CPU E2E and chronological forward CI, record commit SHA and logs.
6. Keep Power Lottery 1,964 missing dates and unverified official CSV provenance visibly flagged until independently resolved.
7. No claims of improved predictive accuracy or production readiness without independent evidence.
