# OF-LGBM: independent OpenFE + LightGBM CPU research

## Boundaries
This standalone system uses its own dependency file and has no imports or E2E connections to V6, CUDA-Q, QAOA, Ising/QUBO, Nemotron, tsfresh, Featuretools, or any retired deep-learning backend. GitHub may show other repository-wide workflows on the same PR; these are not part of standalone OF-LGBM testing.

## Model and optimization
- OpenFE 0.0.12: all 23 operator *families* are executed, with bounded candidate screening. One computed witness of each family is included in LightGBM inputs per view. This is NOT every possible OpenFE formula.
- LightGBM (CPU): three independently trained views and normalized score fusion: recent-5 training anchors 35%, recent-10 anchors 35%, up to 100 distributed historical training anchors 30%. Features for every training row use only its preceding 5 or 10 completed drawings; the larger archive only supplies training samples.
- Exact Python itertools.combinations enumeration of an 8-number pool: 8 choose 6 = 28 for Power and Lotto, 8 choose 5 = 56 for 539. This is exact solely for the eight-number pool and stated score objective.
- Power second zone is a separate 5/10 frequency baseline, not OpenFE/LightGBM.

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
Local data gates check ranges, uniqueness, consecutive sequence numbers, ROC-year rollover, and Lotto/539 dates including ROC-year matches. Power archival CSV lacks dates. SHA-256 verifies bytes, NOT authentic official provenance.
- A successful CI run verifies reproducible execution and limited historical holdout (five per game), NOT full 100/200/500-fold evaluation.
- CSVs are historical but NOT entire exhaustive official archives for all three games. Full official record-by-record provenance remains pending.
- No demonstrated independent prospective predictive edge or improvement over a fair random lottery. Never represent these runs as proof of greater winning probability.

Keep the pull request in DRAFT until the agreed provenance, longer validation, and merge requirements are satisfied.
