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
- Official draw data ingestion: **not yet connected**
- External GitHub crawler source integration: **not yet merged**
- Real-world out-of-sample validation: **not completed**
- Commit history and contributors: use GitHub repository's actual commit and contributor pages
