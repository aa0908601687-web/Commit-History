# OF-LGBM PR #114 post-merge validation trigger

This is a **validation-only** document intentionally matching both OF-LGBM workflow PR path filters. It introduces no production code, feature selection, training, lottery history, or manifest changes.

Pinned base: main commit `0cade19543a8fd60a2f9b76bc7667cd94452645c` (squash merge of PR #114).

Required acceptance:
- OF-LGBM Standalone CPU E2E: three successful Python 3.11 OpenFE 23-operator + pooled LightGBM matrix jobs for power, lotto, and 539.
- OF-LGBM chronological five-refit holdout: three successful matrix jobs for power, lotto, and 539.
- Audit the dependency install steps, full-history installer regression tests, candidate registry status, input snapshot hashes, and uploaded artifacts.
- Do not merge this validation-only PR; it provides a CI trigger/audit record.

Limitations: the 100 candidate configurations are registry-only and not activated as model inputs. No official historical CSV import has occurred. The Power archive has 1,964 missing dates, and there is no established predictive advantage over fair random tickets.

## Fresh verification request — 2026-11-21

Re-run unchanged main-derived OpenFE/LightGBM CPU installation, three-game E2E, and three-game five-refit chronological holdout on this validation branch. No official CSV import, model activation, or production changes. CI pass/fail is determined only from new workflow run IDs and job conclusions.
