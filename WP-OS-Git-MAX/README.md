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
