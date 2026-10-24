# Git 提交與貢獻快取

此功能獨立於開獎頻率快取及選號程式，不改選號設定。

```bash
python3 Git-History-Cache/git_history_cache.py --repo . --repository aa0908601687-web/Commit-History --cache .wp-git-cache/history.json --report-dir git-history-report
```

需完整 Git checkout（Actions 使用 `fetch-depth: 0`）。統計指定 ref（預設 HEAD）所有可達提交，整個儲存庫，不限 WP 資料夾。PR Actions 的 HEAD 可能是測試合併提交，請以產物 `head_sha` 為準。

- `history.json`：提交 SHA、作者名稱／email、作者時間、訊息、變更檔案、各提交新增／刪除行數、二進位檔數；各作者次數／占比／增刪量、UTC 每日趨勢、截止 SHA、產生時間、schema、統計政策及 SHA-256。
- `contributors.svg`：作者提交次數圖。
- `code_changes.svg`：作者新增（綠色）／刪除（紅色）行數圖。

`hit` 表示版本、命名空間、完整性及截止提交一致，直接讀快取且不改寫；`updated` 表示沿用已驗證提交資料並補讀新增 SHA；`refreshed` 表示缺少或損壞快取而重建。分支回退／改寫時只統計當前可達 SHA。舊快取不會在 Git 讀取失敗時被當成最新資料。

新工作流程 **WP-OS Git history cache** 在 push、PR、每天台灣時間08:15及手動觸發時執行。跨次執行用 actions/cache 儲存快取，完成後上傳 `wp-git-history-執行ID` artifact（30天）。下載後可讀 JSON 及 SVG。Actions 不自動提交或推送產物。

統計採原始 Git 作者名稱及 email，不等同 GitHub 帳號或真人身分驗證；作者資料可自填。產物會包含 Git 歷史中已有的作者 email。此統計不做 mailmap 合併。

行數包括所有文字檔（含資料及生成檔），不是純程式碼；不啟用 rename 偵測，改名可能呈現刪除與新增。根提交計入新增，合併提交只計提交次數、不計差異以免重複；二進位檔計檔數，不虛構行數。此政策可能與 GitHub Commit API 的合併統計不同。

SHA-256 可偵測意外損壞或未重算雜湊的變更，不是數位簽章或可信來源認證；請只使用可信分支的快取。快取不提供樂透預測優勢。

測試：`python3 -m unittest discover -s Git-History-Cache -p 'test_git_history_cache.py' -v`
