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

## Status
- GitHub repository linkage: active for this folder
- Native Python prototype: included
- Official draw data ingestion: API updater code exists; scheduled/manual end-to-end success **not yet verified**
- External GitHub crawler source integration: **not yet merged**
- Real-world out-of-sample validation: **not completed**
- Commit history and contributors: use GitHub repository's actual commit and contributor pages


## Automation status (2026-10-21)
- GitHub Actions workflow: [wp-os-git-max.yml](../.github/workflows/wp-os-git-max.yml) committed on `main`; scheduled at 22:00 Asia/Taipei Mondays and Thursdays, and supports manual dispatch.
- On scheduled/manual runs, the workflow attempts to download draw data via `update_official_data.py`, validates the response and existing CSV, runs unit tests and the model, then commits the CSV only if its contents changed.
- **Not yet verified end-to-end**: a successful GitHub Actions run, the external API schema/availability, and a successful bot push. The updater intentionally exits without overwriting saved data if the API response fails validation.
- The source CSV currently contains ten rows ending at draw `115000084`; with only ten rows, the walk-forward backtest is skipped. To perform a real backtest, retain more than ten previous draws in a separate historical dataset.
- Check [Actions](https://github.com/aa0908601687-web/Commit-History/actions) for the latest job logs. In GitHub, select **WP-OS Git MAX data and checks** and **Run workflow** for manual verification.
- External upstream crawlers are not merged; no predictive advantage is established. Results should be checked against the [official historical results site](https://apislb.taiwanlottery.com/lotto/history/history_result/).

## Verification update — 2026-10-21
- Latest publicly verified Power Lottery result: draw `115000084` (2026-10-19), first zone `09 10 21 24 27 36`, second zone `08`. Source: https://lotto.ctbcbank.com/result_all.htm
- `historical_latest_10.csv` already ends at draw `115000084`; no future (unreleased) draw data was added.
- Workflow includes diagnostics for triggering event, commit SHA, CSV row count and latest saved draw; these diagnostics must be read from actual Actions logs to verify execution.
- Successful workflow execution and automated `github-actions[bot]` push have **not** been confirmed. A run without new data does not produce a commit by design.
- Manual check: [WP-OS Git MAX Actions](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-os-git-max.yml) → Run workflow on `main` → examine steps and logs.
- The 5-/10-draw heuristics are not evidence of greater lottery winning probability. Preserve every prediction and source cutoff before its target draw.

## Update status — 2026-10-21 (latest)
- Confirmed latest publicly released draw is `115000084`; the current 10-draw CSV already contains it. No fictional or future results inserted.
- GitHub Actions now uploads the model's output JSON as a 30-day Actions artifact on successful runs, making each run's generated output easier to inspect. This is not the same as a pre-draw prediction timestamp unless execution occurred before the draw.
- Live Actions run results, official API import success, and automated bot push remain **unverified** until inspected from GitHub Actions logs. The GitHub workflow exists, but that alone does not prove it has run successfully.
- Original scoring weights and original stored drawing values are unchanged.
