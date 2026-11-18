# OF-LGBM: independent OpenFE + LightGBM CPU research

## Boundaries
This standalone system uses its own dependency file and has no imports or E2E connections to V6, CUDA-Q, QAOA, Ising/QUBO, Nemotron, tsfresh, Featuretools, or any retired deep-learning backend. GitHub may show other repository-wide workflows on the same PR; these are not part of standalone OF-LGBM testing.

## Model and optimization
- OpenFE 0.0.12: all 23 operator *families* are executed, with bounded candidate screening. One computed witness of each family is included in LightGBM inputs per view. This is NOT every possible OpenFE formula.
- LightGBM (CPU): three independently trained views with causal adaptive score fusion: recent-10, recent-20, and all eligible historical training anchors. Fusion weights are recomputed for each forecast from historical one-step frequency baseline Brier losses, using only already completed draws (up to 30 calibration targets); they are not fixed 35%/35%/30%. These weights are a research heuristic, NOT LightGBM-learned stacking coefficients or validated forecasting gains. Features use only prior completed drawings: 10, 20, or the complete past respectively; no 100-anchor cap. The first 20 draws are warm-up for historical labels.
- Exact Python itertools.combinations enumeration of an 8-number pool: 8 choose 6 = 28 for Power and Lotto, 8 choose 5 = 56 for 539. The scoring now combines **95% mean fused LightGBM per-number score** with **5% shrunk historical two-number co-occurrence**, computed strictly from the last 10 and last 20 earlier draws (10 pseudo-observations per window). This is a transparent *experimental* interaction heuristic, not a validated probability calibration or an increase in winning odds. Enumeration is exact ONLY within the already ranked eight-number pool, never among all possible lottery tickets.
- Power second zone is a separate 10/20 frequency baseline, not OpenFE/LightGBM.

## Real historical CSV snapshots (not synthetic)
- Power: verification/of_lgbm_data/power_history_archive.csv; 1,964 entries through 115000092. A prior in-repository archive, with incomplete official provenance. The most recent 370 entries were separately checked for agreement with the third-party mirror; the 370 overlapped draws matched.
- Lotto: verification/of_lgbm_data/lotto_history_mirror.csv; 412 entries, 112000047 to 115000106, with ISO dates and Lotto special numbers.
- Daily539: verification/of_lgbm_data/539_history_mirror.csv; 1,119 entries, 112000103 to 115000279, with ISO dates.
- Lotto/539 source: public GitHub project oomao/Lottery_Taiwan, commit 12cec5a6995259a54b43f68c29a3e6291f69aba3. The repository describes using Taiwan Lottery official API. It is a third-party mirror, NOT the official publisher; a complete official row-by-row comparison has NOT been performed.
- See verification/of_lgbm_data/SOURCE_MANIFEST.json for pinned source references. Official source reference: https://www.taiwanlottery.com/lotto/history/result_download/ .

## Reproduce Python 3.11 CPU-only validation
Install the OF-LGBM-only requirements:
    python -m pip install -r verification/of_lgbm_requirements.txt

For each game name and matching CSV file above:
    python verification/of_lgbm_real_history_gate.py --game GAME --csv CSV --output source-audit.json
    python verification/of_lgbm_all_operator_e2e.py --game GAME --csv CSV --output model.json
    python verification/of_lgbm_forward_audit.py --game GAME --csv CSV --folds 5 --output forward.json

Run isolated unit tests:
    python -m unittest verification/test_of_lgbm_isolation.py verification/test_of_lgbm_all_operator_e2e.py verification/test_of_lgbm_real_history_gate.py verification/test_of_lgbm_forward_audit.py verification/test_of_lgbm_ticket_selection.py verification/test_of_lgbm_manifest_integrity.py -v

PR GitHub workflows require real CSVs for Power/Lotto/539, explicitly reject synthetic fixtures, and each perform 23 OpenFE-operator + LightGBM training with separate LightGBM refits for five consecutive historical held-out draws.

## Verification limits
Shared CSV gates now check ranges, uniqueness, consecutive draw IDs and ROC-year rollover before both full E2E and forward refits. For datasets with a draw_date column, every row must have a valid, strictly increasing ISO date whose Gregorian year matches the ROC draw ID; missing/partial date values are rejected. Power archival CSV still has **no date column** and is explicitly reported as incomplete (date_coverage=0, date_validation_status=missing_date_column). It must not be described as date-verified. SHA-256 verifies bytes, NOT authentic official provenance.
- A successful CI run verifies reproducible execution and limited historical holdout (five per game), NOT full 100/200/500-fold evaluation.
- CSVs are historical but NOT entire exhaustive official archives for all three games. Full official record-by-record provenance remains pending.
- No demonstrated independent prospective predictive edge or improvement over a fair random lottery. Never represent these runs as proof of greater winning probability.

PR #89 is a separate update branch. Do not merge based only on CI passing: independently verify official draw sources and assess longer out-of-sample results before production use.

## 2026-11-19 research hardening (PR #89)
- Selection logic is deliberately different from earlier published number sets; older tickets cannot be reused as fresh outputs.
- New cross-check unit tests assert deterministic pair-aware search, causal last-10/20 boundaries, nonfinite-score rejection, and incomplete-date rejection.
- Data snapshots appended (Power 115000092; Lotto 115000106; 539 115000279), with non-official provenance recorded in SOURCE_MANIFEST. Source authentication and longer nested walk-forward remain outstanding.
- CI passing does not validate empirical winning probability. Neither the heuristic pair weight nor 10 pseudo-observations were learned from independent test data.

- A new manifest integrity regression test compares each CSV row count and cutoff with SOURCE_MANIFEST, explicitly flags missing Power dates, and prevents labeling these third-party snapshots as officially authenticated.

## 2026-11-19 data-integrity hardening
- SOURCE_MANIFEST pins the **actual SHA-256** of each of the three current repository CSV snapshots, taken from the successful main-branch E2E execution (run 44367099716). CI recomputes the SHA-256 from CSV bytes and fails on a mismatch, missing rows, changed cutoff, invalid chronological structure or unexpected dataset path.
- The check distinguishes structural/date-format integrity from source authenticity. **Neither matching SHA-256 nor passing CI proves the draws were copied accurately from Taiwan Lottery.** Official full row-by-row comparison has not been completed.
- Power has 1,964 historical records with **no draw_date field at all**. The audit explicitly retains date_coverage=0 and missing_all_dates; do NOT infer dates from sequential draw IDs or fill them from assumed weekday schedules.
- Lotto and 539 have structurally consistent ISO dates, but dates and numbers have not been authenticated against all official rows.
- The official Taiwan Lottery yearly download entry point is https://www.taiwanlottery.com/lotto/history/result_download/ ; official files have **not** been imported into this repository as part of this change. The currently exposed yearly files may be updated monthly and may not cover the most recent draws.
- If a CSV must be updated, independently inspect the underlying source, update its manifest digest deliberately, run the full validation gate and E2E, and preserve unverified provenance until a documented actual official comparison is completed.

## Prediction-error reporting hardening (2026-11-19)
- The five-fold true ML holdout now reports the *exact* 49/38/39-select-k hypergeometric distribution of matched first-zone numbers, fair expected mean matches, the observed mean, expected 3+ events, an exact one-sided binomial fair-null tail, and a Wilson 95% interval for the observed 3+ rate.
- Power separately reports second-zone matches, exact 1/8 fair baseline, expected count, Wilson interval and binomial tail. **The second-zone pick remains a frequency baseline and is not a trained LightGBM output.**
- The older one-seed random ticket remains for reproducible illustration but is explicitly labeled insufficient as evidence of prediction edge. A five-draw comparison is very small; no statistical power or search-corrected superiority is inferred. The report always labels `statistically_verified_predictive_advantage=false`.
- This is a **metric/evaluation fix**, not a claim to have repaired the intrinsic uncertainty of fair lottery draws. Model fusion weights, pair heuristic and ticket selection stay unchanged to avoid post-hoc tuning to five observed outcomes.

## 2026-11-19 adaptive three-view research branch
- Replaces the prior fixed three-view score-fusion weights with causally calibrated weights computed from preceding draws only; first 20 draws are warmup.
- The calibration is based on smoothed single-number historical-frequency Brier scores for each view; it must not be described as trained LightGBM meta-stacking or proven to improve accuracy.
- Existing official-source gaps and Power date coverage=0 remain unresolved. The real-history gates, separate E2E workflows, and five-fold forward holdout must pass before merge.

## Power draw-date remediation gate (research PR, 2026-11-19)
- Current archived Power CSV has 1,964 entries **without any date column**. The archive remains untouched until official date evidence is actually imported. No weekdays, draw-id interpolation, or guessed dates are allowed.
- Official reference: https://www.taiwanlottery.com/lotto/history/result_download/ (yearly official exports; the current-month rows may not be included in the monthly downloads).
- Independently obtain the original official files and review their origin. Normalize to a separate CSV with header `draw_id,draw_date,first,second` (ISO date) without altering original downloaded files.
- Run `python verification/of_lgbm_power_date_reconcile.py --archive verification/of_lgbm_data/power_history_archive.csv --dated-source PATH_TO_INDEPENDENTLY_REVIEWED_DATES.csv --output power_history_with_dates.csv --report power_date_reconciliation.json`.
- Import refuses missing/extra draw IDs, any mismatch in both zones' exact winning numbers, duplicate IDs, invalid date or ROC year, and dates not strictly increasing. It writes a **new** CSV and audit hashes only after all records match, never replacing the archived source in place.
- The importer cannot certify a source as official by itself; the report always keeps `independent_official_provenance_verified=false`. An independent official record-by-record audit and explicit source log must occur before changing that flag.
- Unit tests for this reconciliation run within both standalone E2E and five-fold audit CI. These tests **do not** mean dates have already been recovered. The existing snapshot manifest continues to report date_coverage=0 and official provenance incomplete until real official data has been checked and imported.

### Power full-date gap verification — 2026-11-19
- Date availability remains **0 of 1,964** in the pinned repository Power CSV (all 1,964 missing). Do not interpret this as missing draw numbers or completed remediation.
- Taiwan Lottery's official annual data download page lists a draw-date field: https://www.taiwanlottery.com/lotto/history/result_download/ . The official page says files are updated on the **5th of each month through the previous month**, so its current-year export may omit November draws. The official files have NOT been downloaded or independently compared in this repository update.
- Date reconciliation now checks *both* draw-ID ordering and chronological date ordering against the archive, even when historical winning numbers happen to be identical across two records. Fail closed on all mismatch, no source overwrite.
- Existing PR tests and E2E cover reconciliation correctness with synthetic fixtures, not actual 1,964-draw official date recovery. Retain `missing_all_dates`, `official_row_verification=not_completed` and existing pinned SHA-256 until certified imports are available.

### Historical completeness disclosure (2026-11-19)
Run `python verification/of_lgbm_history_completeness.py --output history-completeness.json` to produce an independent, fail-closed, three-game local archive snapshot inventory. Both standalone and forward-test CI execute it and attach JSON evidence. It checks original CSV hashes, declared row count/cutoff, legal sequential in-file draw IDs, date availability and manifest source flags, but **does not certify that the archive includes the entire official history or the latest official draw**.
Pinned snapshot: Power 1,964 draws, 0 dates / 1,964 missing, last 115000092; Lotto 412 draws, dates format-consistent but not officially authenticated, last 115000106; 539 1,119 draws, dates format-consistent but not officially authenticated, last 115000279. The exact snapshot may become stale; this workflow intentionally fails closed rather than inventing new dates or modifying existing draw numbers.
Official annual-download reference: https://www.taiwanlottery.com/lotto/history/result_download/ . Full official file import, row-level verification, and comparison to the latest published result remain outstanding before a true full-history certification can be claimed.
