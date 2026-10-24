# Git 提交與貢獻快取

此功能獨立於開獎頻率快取。原有歷史開獎選號模式不變，另提供 Git 快取選號模式。

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

## Git 快取選號

```bash
python3 Git-History-Cache/git_history_cache.py --repo . --repository aa0908601687-web/Commit-History --cache .wp-git-cache/history.json --report-dir git-history-report --pick
```

先核對目前 Git HEAD、更新並驗證快取，再將截止 Commit、所有提交記錄、貢獻者統計及新增／刪除行數的底層數據一起做 SHA-256，轉成固定種子。圖表是同一份統計的呈現，不解析 SVG 圖片。排除快取產生時間，避免同一提交只因重新建立快取而換號。

SHA-256計數器搭配拒絕抽樣及部分 Fisher–Yates 洗牌，產生第一區01～38中6個不重複號碼及第二區01～08中1碼。演算法固定版本 `wp-git-seeded-sha256-fisher-yates-v1`；相同版本、repository、截止Commit及資料得到相同號碼。新提交改變種子，未保證新號碼一定不同。

Actions 現在會將 `git_pick.json` 一起上傳至 `wp-git-history-執行ID` artifact，包含號碼、截止Commit、來源快取雜湊、種子及演算法版本。離線函式 `pick_from_cache` 也要求指定repository和預期截止Commit，拒絕不符或損壞的快取；CLI會先更新至指定ref，離線函式不自行查詢GitHub最新狀態。

此模式不使用開獎資料，不與歷史頻率模型混合、不代表中獎機率提高，也不能把可修改的Git資料當成公平公開抽籤。更動提交可以改變種子；若用於開獎後比對，需開獎前另行封存完整輸入和結果。快取SHA-256不是數位簽章。
