# WP-OS Git MAX

Independent Taiwan Power Lottery research prototype. This repository folder is not an official lottery predictor and does **not** contain externally merged upstream source code.

## Rules
- First zone: choose 6 distinct numbers from 01–38
- Second zone: choose 1 number from 01–08
- Prediction features use only the last **5 and 10 completed draws**
- The two zones are analyzed separately
- All rules and weights below are **initial research choices, not validated predictive advantages**

## Data format
CSV must contain `draw_id,first,second`. The `first` field contains six space-separated integers, e.g. `01 03 08 17 29 38`. Rows are oldest to newest. No special draw is included in the six first-zone numbers.

## Commands
```bash
python3 WP-OS-Git-MAX/wp_os_git_max.py --csv historical.csv
python3 WP-OS-Git-MAX/wp_os_git_max.py --csv historical.csv --backtest
python3 -m unittest discover -s WP-OS-Git-MAX -p 'test_*.py'
```

## Model
First zone: 5-draw frequency 45%, 10-draw frequency 25%, Bayesian-smoothed 10-draw frequency 15%, last-draw repetition 10%, overdue recency 5%. Second zone uses its own counts and the same fixed weights. No second-zone/first-zone coupling is assumed. Two-number co-occurrence is reported for research, but not used in the final pick.

The model does not claim an advantage over random selection. A fair first-zone jackpot with matching second zone has probability 1 / (C(38,6)*8) = 1 / 22,085,448. SHA-256 log hashes confirm file integrity only, not that a forecast predates a draw. Public GitHub commits give an independent timestamp for committed forecasts.

## Verified status — 2026-10-22 (Asia/Taipei)
- Repository main and Actions are accessible. Audited source commit: d8cfb0ad8b93fecb41c2c195a8875958e2d99983.
- Scheduled run 39796734408 completed successfully at 22:34 Taiwan time. Official API updater returned `No changes; through draw 115000085`; unit tests and prediction artifact upload succeeded.
- Run 39796777590 validated the latest archived prediction commit successfully.
- The scheduled run reported `No updated draw data; no commit needed.` This verifies the no-change path, **not** a successful automated bot push.
- Latest CSV contains draws 115000076–115000085. Last result: first zone 01 09 10 21 22 31, second zone 06.
- Reproduced candidate for draw 115000086: first zone 01 09 10 21 22 27, second zone 06. Output hash matches the committed prediction.
- This update retains a separate historical_archive.csv for future walk-forward testing while feature windows remain strictly 5/10 draws. The initial archive has only ten existing records; no historical results are invented. Real-data backtesting requires at least eleven records.
- Tests now verify each walk-forward call receives only earlier draws, earlier-than-ten records do not change a prediction, and the CSV loader rejects reversed order.
- The model uses identical 45/25/15/10/5 weights for both zones. Two-number co-occurrence is reported but does not affect selection. This describes the actual implementation.
- Each of the ten stored draws has not been independently rechecked against official raw records in this audit. Successful ingestion alone is not complete independent data certification.
- External crawler code is not merged. Predictive advantage and real-world out-of-sample performance remain unestablished.
- Schedule: Mondays and Thursdays 22:00 Asia/Taipei, subject to GitHub queue delays; manual dispatch also supported.

[Scheduled run](https://github.com/aa0908601687-web/Commit-History/actions/runs/39796734408)

## External draw verification — 2026-10-23 (Asia/Taipei)
- Independently checked the latest stored draw, **115000085 (2026-10-22)**, against the Taiwan Lottery results displayed by CTBC: first zone `01 09 10 21 22 31`, second zone `06`.
- Official result page: https://lotto.ctbcbank.com/result_all.htm (live page may change after the next draw).
- Secondary dated report: https://www.cna.com.tw/news/ahel/202610220397.aspx .
- The repository's `historical_latest_10.csv` and `historical_archive.csv` both retain the same latest draw. No upcoming draw is added before its result is published.
- **Verification scope:** only draw 115000085 was externally cross-checked in this update; the prior nine draws were not independently rechecked. Existing tests and Actions were not executed by this documentation-only update; prior README claims about workflow runs are historical, not fresh run attestations. Automated bot push success remains unproven unless a bot-authored data commit is observed.

## Storage policy — 2026-10-26
- `historical_latest_20.csv` retains the most recent 20 available verified draws and is the workflow input.
- Selection and feature caches continue to use only the last 5/10 draws; weights and model version are unchanged.
- `historical_latest_10.csv` is maintained for compatibility with existing snapshots.
- `historical_archive.csv` retains the complete verified archive for walk-forward evaluation.

## Scion-style three-layer cache V3

This project uses a **Scion-style** cache layout; it is not Anthropic's internal Scion implementation.

1. **Node layer:** `.cache/wp-os-v3` inside the current GitHub Actions runner.
2. **Regional layer:** GitHub Actions cache restores validated feature and prediction objects across workflow runs.
3. **Authoritative layer:** repository CSV files and Python source are the source of truth.

The cache key includes the draw CSV and cache/model source files. Each cached object also carries its own schema/model identity and SHA-256 digest. A missing, stale, malformed, or modified object is recomputed, and the workflow compares cached output with direct computation before publishing artifacts. This architecture improves reuse and integrity; it does not improve lottery odds or imply Anthropic compatibility.


## Independent 10/20-draw control

The production research selection remains the fixed 5/10-draw model. An independent 10/20-draw control model runs beside it using the same data cutoff and the same weight categories scaled to 10/20 draws. The workflow publishes both control output and a main-versus-control overlap report in the `wp-os-control-10-20-<run_id>` artifact.

The control is for prospective comparison only. It does not feed the production selection, the V3 prediction cache, or contributor statistics, and it must not replace the main model without sufficient pre-draw observations and walk-forward evidence.


## Equal-weight 5/10 + 10/20 fusion

The fusion model combines the complete score tables of the main 5/10 model and the independent 10/20 control at **50% each**. It emits one fused first-zone six-number set and one second-zone number while preserving both component predictions in the JSON output.

The fusion needs at least 20 completed draws. Walk-forward evaluation starts with the 21st stored draw and uses only information available before each target draw. The workflow publishes the current fusion output and its historical backtest in the `wp-os-fusion-5-10-20-<run_id>` artifact. This is a research comparison and does not increase the theoretical lottery odds.


## V3 cached control and fusion results

The independent 10/20 control and equal-weight 5/10 + 10/20 fusion are now connected to the Scion-style three-layer cache V3. Their cache identities include the latest 20-draw history digest, cache schema, and model version. Every workflow performs a second cached run and compares each cached prediction with a fresh direct computation before publishing it. Any changed data, model source, malformed JSON, digest mismatch, or cutoff mismatch causes automatic recomputation.


## V3 full-history backtest cache

The complete `historical_archive.csv` now has a versioned full-history backtest cache covering the main 5/10, control 10/20, and equal-weight fusion models. Its identity includes the complete ordered archive SHA-256, first and latest draw IDs, draw count, schema version, and all three model versions. A changed archive, changed model version, malformed cache, missing model result, or digest mismatch forces a complete walk-forward recomputation. The second workflow pass must report a cache hit and identical backtest payload before publication.


## Open-source provenance and geographic scope

- **Lottery jurisdiction:** This research prototype targets Taiwan's Power Lottery (威力彩). Its geographic scope does **not** establish the nationality or location of the software authors.
- **Repository implementation:** WP-OS Git MAX is an independently maintained research prototype; this documentation does not claim that external upstream lottery projects' source code was merged into the prediction engine.
- **Related public reference projects:** [ky921d/mk](https://github.com/ky921d/mk) and [stu01509/TaiwanLotteryCrawler](https://github.com/stu01509/TaiwanLotteryCrawler) are Taiwan-lottery-oriented public repositories. Their maintainers' nationalities have **not** been independently verified, and references to them should not be interpreted as a confirmed code dependency or an upstream integration.
- **Platform and tools:** GitHub/GitHub Actions provide repository hosting and workflow automation; Python is a programming language. Neither determines the nationality of the project's developer.
- **Verification boundary:** This section documents provenance distinctions only, not live API connectivity, installed upstream packages, official lottery affiliation, or improved prediction accuracy.


## Fourth independent research model — Adaptive Bayesian Ensemble MAX (2026-10-28)

This **optional fourth model** does not replace the existing 5/10 main model, 10/20 control, or fixed 50:50 fusion, and is not wired into the existing V3 cache. It has its own workflow, tests, predictions and optional backtest artifacts.

- Source: [adaptive_bayesian_ensemble_max.py](adaptive_bayesian_ensemble_max.py). Command: `python WP-OS-Git-MAX/adaptive_bayesian_ensemble_max.py --csv WP-OS-Git-MAX/historical_archive.csv`. Use `--backtest` for sequential historical evaluation.
- Requires at least 20 completed prior draws. The 5/10 and 10/20 score tables are combined with separate zone-specific adaptive weights.
- Weight updates use *only earlier draws* and prequential Brier-like score losses. A conservative equal-weight prior, 10-observation warmup and shrinkage prevent abrupt weight shifts; weights are heuristics, not calibrated posterior probabilities. This is **not a full Bayesian posterior or trained stacking neural network**.
- Workflow: [WP-OS Adaptive Bayesian Ensemble MAX](../.github/workflows/wp-os-adaptive-ensemble.yml) runs isolated tests and, when sufficient archive data exists, produces JSON output and a prequential backtest; absent data is explicitly reported as skipped.
- Validated source integration, successful remote Actions executions, actual sample-out performance and predictive advantage must be checked separately. No statistical evidence is claimed here of improved lottery odds.
