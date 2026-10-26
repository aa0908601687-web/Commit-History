# Commit-History

Unified overview for the independent WP-OS Git MAX research project.

[![WP-OS Git MAX](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-os-git-max.yml/badge.svg)](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-os-git-max.yml)
[![Commit and contributor sync](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-contributors-sync.yml/badge.svg)](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-contributors-sync.yml)
[![Test coverage](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-coverage.yml/badge.svg)](https://github.com/aa0908601687-web/Commit-History/actions/workflows/wp-coverage.yml)

## System overview

| Area | Purpose | Entry point | Output |
| --- | --- | --- | --- |
| WP-OS Git MAX | Data checks, deterministic 5/10-draw research selection, backtest and three-layer cache verification | [Model documentation](WP-OS-Git-MAX/README.md) · [Workflow](../../actions/workflows/wp-os-git-max.yml) | Prediction and V3 cache artifacts |
| Commit and contributor audit | Commit history, contributor counts, daily activity, additions and deletions | [Workflow](../../actions/workflows/wp-contributors-sync.yml) | `wp-contributors-report` artifact |
| Test coverage | Automated Python test coverage | [Workflow](../../actions/workflows/wp-coverage.yml) | Coverage status and report |

## Separation guarantee

The two workflows share this overview only. Commit counts, contributor identities, and code additions/deletions are **not** lottery-selection features and do not enter the prediction cache. A contributor-report failure does not change a selection result, and selection/cache data does not change contributor attribution.

## Contributor report contents

The latest successful `WP-OS commit and contributor sync` run publishes one `wp-contributors-report` artifact containing:

- `report.json` and `contributors.csv`
- `contributors.svg`
- `code_changes.json`
- `code_changes.svg`
- `code_changes_by_author.csv`
- `code_changes_by_day.csv`

Artifacts are retained by GitHub Actions for 30 days.

## Research notice

This project is educational. Cache integrity, reproducibility, commit history, and code-contribution statistics do not establish a predictive advantage for a fair lottery.
