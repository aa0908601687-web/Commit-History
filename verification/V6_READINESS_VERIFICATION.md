# V6 OF3Q independent readiness and verification boundaries

This change adds **read-only engineering evidence**, not an automatic predictor improvement.

## What is actually exercised

1. **Existing unified CPU end-to-end** (`.github/workflows/v6-of3q-unified-e2e.yml`) runs the actual OpenFE + LightGBM three-view model, optional CUDA-Q local CPU simulator, Ising/QUBO CPU reference and 8-choose-6 exact search, plus separate offline Rocq / structured-data checks. Its exit status and artifacts must be verified on the **specific latest main SHA**.
2. **Research machine-learning smoke test** (`.github/workflows/v6-readiness-validation.yml`) performs **one** genuine train-before-reveal OpenFE + LightGBM holdout comparison of the baseline and expanded features. One draw is NOT evidence of predictive superiority.
3. **Causal 100/200/500 feature-only reference** (`verification/v6_readiness_audit.py`) scans only the ten most recently completed draws for each target. It runs a fixed, lightweight, reproducible **CPU formula, not OpenFE retraining**, against fair exact combinatorial probabilities, seeded uniform-random tickets and a fixed ticket. The formula's weights are fixed before the holdout; no holdout tuning or automatic production promotion takes place.
4. **Historical CSV validation** checks columns, 38/6+8/1 limits, duplicate draws, number uniqueness, monotonically increasing draw IDs, completeness of parseable records and SHA-256 integrity. This CSV **has no date column**. Neither official second-source parity nor draw-calendar completeness is presently proven.
5. **Coverage** is separated: the existing `wp-coverage.yml` remains a **100% threshold for its WP-OS-Git-MAX source boundary** and is now triggered by changes to the research code. The new workflow emits actual statement+branch coverage of the **two research modules** and requires a separate 100% statement-and-branch coverage gate. This threshold applies to these measured modules only, not the whole repository. Do not label the combined repository 100% until all intended modules are measurably included and all branches tested.

## Explicit exclusions / missing evidence

- No physical QPU or paid/cloud hardware dependency. Retired optimizer and retired ML backends stay excluded.
- No authorized live external model inference or deployed interactive prediction service has been confirmed; the Nemotron extraction check in the older workflow is **offline**.
- Rocq candidate-invariant compilation is **not** a proof of the Python model's numerical correctness or that future lottery outcomes can be forecast.
- The lightweight reference 100/200/500 is **not equivalent** to a full model retraining walk-forward. A full 100/200/500 OpenFE refit remains unverified; run it only with a practical resource budget and separate leakage-controlled scoring before claiming that milestone.
- Site snapshots may experience GitHub API 403 errors. Successful direct connected GitHub read calls are independent evidence but do not repair the Site service permissions.

## Independent validation / evidence links

- Current repository: https://github.com/aa0908601687-web/Commit-History
- Latest main workflow list: https://github.com/aa0908601687-web/Commit-History/actions
- Each GitHub Actions artifact records its source commit. Validate `main` and artifact SHA match before marking a gate passed.
- Use `v6-readiness-audit.json`, `v6-real-openfe-smoke.json`, and `v6-research-coverage.json` to distinguish reference, real model, and test completeness.

**No fair lottery algorithm has been demonstrated here to increase the theoretical single-ticket winning probability.**
