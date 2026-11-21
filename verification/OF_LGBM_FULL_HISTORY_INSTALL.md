# OF-LGBM full-history staging and installation (safety gate)

**Current status (2026-11-20): installation tool and tests added in PR #109, but original official yearly exports have NOT been imported or validated.** This is not proof of official completeness.

The live main snapshots are historical research snapshots, not certified all-time official archives. The Power snapshot contains **1,964 draw rows without dates**. Do not infer dates from draw IDs or a calendar. Keep `SOURCE_MANIFEST.json` pinned to the unchanged snapshots until authentic documents have been independently reviewed.

## Official primary source

Taiwan Lottery yearly downloads: https://www.taiwanlottery.com/lotto/history/result_download/ . Their annual files may lag current-month draws. Retrieve the actual original files, keep them unchanged, record URLs, original SHA-256, download dates, the covered years, and an independent source review. Never turn a third-party mirror into a claim of official certification.

## Install/check locally (Python 3.11)

```bash
python -m pip install --disable-pip-version-check -r verification/of_lgbm_requirements.txt
python -m unittest verification/test_of_lgbm_full_history_install.py -v
python verification/of_lgbm_full_history_install.py \
  --source-dir /secure/reviewed_normalized_sources \
  --snapshot-dir verification/of_lgbm_data \
  --stage-dir /secure/of_lgbm_stage_20261120
```

Normalize independently obtained source files into `power.csv` (`draw_id,draw_date,first,second`), `lotto.csv` (`draw_id,draw_date,first,special`) and `539.csv` (`draw_id,draw_date,first`). Dates must be ISO YYYY-MM-DD, IDs nine-digit ROC period IDs, first-zone numbers separated by spaces, and source rows in increasing draw order. The command refuses missing dates, duplicate IDs, nonconsecutive draws, overlapping numbers/dates that disagree with existing snapshots, and an existing destination. A valid staging output creates three **separate** CSV copies and `STAGING_AUDIT.json` with fingerprints.

The staged sources are **quarantined candidate imports**, never a main-branch data replacement. CSV structural validation and equal overlapping rows do not prove that the source is official, exhaustive since game inception, or current through today. All output attestation fields therefore remain false.

## Follow-up gates before production update

1. Independently authenticate primary original annual files and their provenance, verify all years/periods and current-month live draws using officially published results, and keep discrepancy review records.
2. Ensure the Power date recovery is supported by actual official records for **all** 1,964 undated rows; never fabricate.
3. After explicit approved source review, stage/compare all three games; in a separate PR update canonical CSVs, manifest SHA-256 and coverage assertions together.
4. Rerun the real-data three-game OF-LGBM OpenFE/LightGBM E2E, five-fold sample-out audit, and longer prospective evaluations. Passing tests does not prove a predictive advantage.

PR #109 is installation **readiness**, not completion of these gates.
