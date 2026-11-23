# Official lottery data import (isolated integration)
Official download page: https://www.taiwanlottery.com/lotto/history/result_download/

This directory is separate from WP-OS Git MAX and does not alter existing selection logic.

1. Download annual historical files from Taiwan Lottery.
2. Convert official spreadsheet files to CSV without changing values, if needed.
3. Confirm column names: 遊戲名稱,期別,開獎日期,獎號1..獎號6,特別號.
4. Run: `python3 official_lottery_import/import_csv.py <files...> --out output/official`.
5. Inspect manifest SHA-256 and row counts; independently cross-check against the official website before marking records verified.

The importer rejects duplicate issues, illegal numbers and missing issue/date. It does NOT download files, authenticate source provenance, train OpenFE/LightGBM, or establish E2E production readiness. The official page says updates are monthly; recent draws may require separate acquisition.

## Official source verification (2026-11-22)

Taiwan Lottery's download page lists annual files for ROC years 96–115 and documents the columns: 遊戲名稱, 期別, 開獎日期, 銷售總額, 銷售注數, 總獎金, 獎號1..6, 特別號. Its published update schedule is the fifth day of each month through the previous month. Therefore the 115-year annual download alone cannot be assumed to contain November 2026 draws.

Do not label the dataset 'officially verified' until the downloaded source files, dates, counts and hashes have been checked. The importer currently expects CSV; actual download file format and game-specific schema need validation before running a production import. The government open-data catalog also links the same public-data category: https://data.gov.tw/dataset/72921 .
