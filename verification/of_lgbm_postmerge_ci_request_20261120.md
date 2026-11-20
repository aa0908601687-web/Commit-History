# OF-LGBM post-merge validation request — 2026-11-20

Baseline main commit: `b206f8e67d49c9a9a0da4b84f28ed7a56fbb7996` (PR #105 squash merge).

This marker exists only to trigger the existing PR-scoped OF-LGBM standalone three-game CPU E2E and five-fold real-ML chronological holdout workflows. It does not modify model code, selection logic, original CSV snapshots, SOURCE_MANIFEST, SHA-256 digests, or historical draw dates. **Do not merge** this validation-only PR.

Acceptance: power/lotto/539 matrix jobs must all succeed for both workflows. Keep Power 1,964 missing draw dates, incomplete official record-by-record source verification, non-certification of up-to-date completeness, and no demonstrated advantage over fair random selection explicit. Any workflow success is execution evidence only, not official attestation or proof of predictive accuracy.
