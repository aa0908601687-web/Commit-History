# OF-LGBM: independent OpenFE + LightGBM CPU research

## Boundaries
This standalone system uses its own dependency file and has no imports or E2E connections to V6, CUDA-Q, QAOA, Ising/QUBO, Nemotron, tsfresh, Featuretools, or any retired deep-learning backend. GitHub may show other repository-wide workflows on the same PR; these are not part of standalone OF-LGBM testing.

## Model and optimization
- OpenFE 0.0.12: all 23 operator *families* are executed, with bounded candidate screening. One computed witness of each family is included in LightGBM inputs per view. This is NOT every possible OpenFE formula.
- LightGBM (CPU): three independently trained views and normalized score fusion: recent-10 training anchors 35%, recent-20 anchors 35%, all eligible historical training anchors 30%. Features use only prior completed drawings: 10, 20, or the complete past respectively; no 100-anchor cap. The first 20 draws are warm-up for historical labels.
- Exact Python itertools.combinations enumeration of an 8-number pool: 8 choose 6 = 28 for Power and Lotto, 8 choose 5 = 56 for 539. The scoring now combines **95% mean fused LightGBM per-number score** with **5% shrunk historical two-number co-occurrence**, computed strictly from the last 10 and last 20 earlier draws (10 pseudo-observations per window). This is a transparent *experimental* interaction heuristic, not a validated probability calibration or an increase in winning odds. Enumeration is exact ONLY within the already ranked eight-number pool, never among all possible lottery tickets.
- Power second zone is a separate 10/20 frequency baseline, not OpenFE/LightGBM.

## Real historical CSV snapshots (not synthetic)
- Power: verification/of_lgbm_data/power_history_archive.csv; 1,963 entries through 115000091. A prior in-repository archive, with incomplete official provenance. The most recent 370 entries were separately checked for agreement with the third-party mirror; the 370 overlapped draws matched.
- Lotto: verification/of_lgbm_data/lotto_history_mirror.csv; 411 entries, 112000047 to 115000105, with ISO dates and Lotto special numbers.
- Daily539: verification/of_lgbm_data/539_history_mirror.csv; 1,116 entries, 112000103 to 115000276, with ISO dates.
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
    python -m unittest verification/test_of_lgbm_isolation.py verification/test_of_lgbm_all_operator_e2e.py verification/test_of_lgbm_real_history_gate.py verification/test_of_lgbm_forward_audit.py -v

PR GitHub workflows require real CSVs for Power/Lotto/539, explicitly reject synthetic fixtures, and each perform 23 OpenFE-operator + LightGBM training with separate LightGBM refits for five consecutive historical held-out draws.

## Verification limits
Shared CSV gates now check ranges, uniqueness, consecutive draw IDs and ROC-year rollover before both full E2E and forward refits. For datasets with a draw_date column, every row must have a valid, strictly increasing ISO date whose Gregorian year matches the ROC draw ID; missing/partial date values are rejected. Power archival CSV still has **no date column** and is explicitly reported as incomplete (date_coverage=0, date_validation_status=missing_date_column). It must not be described as date-verified. SHA-256 verifies bytes, NOT authentic official provenance.
- A successful CI run verifies reproducible execution and limited historical holdout (five per game), NOT full 100/200/500-fold evaluation.
- CSVs are historical but NOT entire exhaustive official archives for all three games. Full official record-by-record provenance remains pending.
- No demonstrated independent prospective predictive edge or improvement over a fair random lottery. Never represent these runs as proof of greater winning probability.

The previous PR #77 was merged. Keep the new 10/20/full-history change in a separate draft PR pending successful isolated CI, provenance review and longer validation.

## 2026-11-19 research hardening (PR #89)
- Selection logic is deliberately different from earlier published number sets; older tickets cannot be reused as fresh outputs.
- New cross-check unit tests assert deterministic pair-aware search, causal last-10/20 boundaries, nonfinite-score rejection, and incomplete-date rejection.
- Data snapshots appended (Power 115000092; Lotto 115000106; 539 115000279), with non-official provenance recorded in SOURCE_MANIFEST. Source authentication and longer nested walk-forward remain outstanding.
- CI passing does not validate empirical winning probability. Neither the heuristic pair weight nor 10 pseudo-observations were learned from independent test data.
