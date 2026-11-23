#!/usr/bin/env python3
"""Import locally downloaded Taiwan Lottery official CSV files; no prediction claims."""
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

GAMES = {"威力彩": (6, 38), "大樂透": (6, 49), "今彩539": (5, 39)}
ALIASES = {"威力彩": "威力彩", "大樂透": "大樂透", "今彩539": "今彩539"}
def load_csv(path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp950", "big5"):
        try:
            text = raw.decode(enc)
            return list(csv.DictReader(text.splitlines())), hashlib.sha256(raw).hexdigest(), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Unsupported CSV encoding")
def date_from_source(value):
    """Validate actual Gregorian draw dates in YYYY-MM-DD or YYYY/MM/DD."""
    cleaned = value.replace("/", "-")
    if len(cleaned) != 10 or cleaned[4] != "-" or cleaned[7] != "-":
        raise ValueError("expected YYYY-MM-DD or YYYY/MM/DD")
    return date.fromisoformat(cleaned)

def normalize(rows):
    result = {"威力彩": [], "大樂透": [], "今彩539": []}
    seen = set()
    for i, row in enumerate(rows, 2):
        row = {str(k).strip(): (v or "").strip() for k, v in row.items() if k is not None}
        game = ALIASES.get(row.get("遊戲名稱", ""))
        if game is None:
            continue
        count, maximum = GAMES[game]
        issue = row.get("期別", "")
        date = row.get("開獎日期", "")
        if not issue or not date:
            raise ValueError(f"row {i}: missing issue/date")
        try:
            date_from_source(date)
        except ValueError as exc:
            raise ValueError(f"row {i}: invalid draw date {date!r}") from exc
        try:
            nums = [int(row[f"獎號{j}"]) for j in range(1, count + 1)]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"row {i}: invalid numbers") from exc
        if len(set(nums)) != count or any(not 1 <= n <= maximum for n in nums):
            raise ValueError(f"row {i}: out-of-range or duplicate numbers")
        extra = row.get("特別號", "")
        if game == "威力彩":
            try:
                second = int(extra)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"row {i}: invalid second-area number") from exc
            if not 1 <= second <= 8:
                raise ValueError(f"row {i}: invalid second-area number")
        elif game == "大樂透" and extra:
            try:
                special = int(extra)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"row {i}: invalid special number") from exc
            if not 1 <= special <= 49 or special in nums:
                raise ValueError(f"row {i}: invalid special number")
        key = (game, issue)
        if key in seen:
            raise ValueError(f"row {i}: duplicate game/issue")
        seen.add(key)
        result[game].append({"game": game, "issue": issue, "date": date,
                             "numbers": nums, "special": extra})
    return result
def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path, help="Official downloaded CSV files")
    p.add_argument("--out", type=Path, default=Path("output/official"))
    args = p.parse_args()
    all_rows, sources = [], []
    for file in args.files:
        rows, digest, enc = load_csv(file)
        all_rows.extend(rows)
        sources.append({"file": str(file), "sha256": digest, "encoding": enc, "rows": len(rows)})
    grouped = normalize(all_rows)
    if not any(grouped.values()):
        raise SystemExit("No supported game rows; inspect official source schema")
    args.out.mkdir(parents=True, exist_ok=True)
    for game, entries in grouped.items():
        with (args.out / (game + ".json")).open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    manifest = {"source": "user-supplied files; official provenance requires independent verification",
                "sources": sources, "counts": {k: len(v) for k, v in grouped.items()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
