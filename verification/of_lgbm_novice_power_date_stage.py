#!/usr/bin/env python3
"""Stage third-party Power draw dates for review; never certify official provenance."""
import argparse
import csv
import hashlib
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "https://lotto.novice.tw/game/super638"
FIELDS = ("draw_id", "draw_date", "first", "second")

class TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None
    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = []
    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append(" ".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

def download_page(page):
    url = SOURCE + ("?y=" + str(page[0]) + "&p=" + str(page[1]) if isinstance(page, tuple) else "?p=" + str(page))
    req = Request(url, headers={"User-Agent":"OF-LGBM-history-audit/1.0 (public research; no automatic publication)"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")

def rows_from_html(html):
    p = TableRows()
    p.feed(html)
    out = []
    for cells in p.rows:
        if len(cells) < 3:
            continue
        ident = cells[0].strip()
        day = cells[1].strip()
        numbers = re.findall(r"(?<!\d)\d{1,2}(?!\d)", cells[2])
        if re.fullmatch(r"\d{8,9}", ident) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            ident = ident.zfill(9)
            if len(numbers) != 7:
                raise ValueError(f"unexpected number format for {ident}: {cells[2]!r}")
            first = tuple(sorted(map(int, numbers[:6])))
            second = int(numbers[6])
            if len(set(first)) != 6 or min(first) < 1 or max(first) > 38 or not 1 <= second <= 8:
                raise ValueError(f"invalid Power numbers {ident}")
            out.append((ident, day, first, second))
    return out

def collect(fetch=download_page, max_pages=300):
    found = {}
    for page in range(1, max_pages + 1):
        rows = rows_from_html(fetch(page))
        if not rows:
            if page == 1:
                raise ValueError("no draw rows on first page; source format may have changed")
            break
        new = 0
        for ident, day, first, second in rows:
            value = (day, first, second)
            if ident in found and found[ident] != value:
                raise ValueError(f"conflicting source entries for {ident}")
            if ident not in found:
                found[ident] = value
                new += 1
        if new == 0:
            break
    else:
        raise ValueError("page cap reached: possibly incomplete source")
    # Historical 2008-2010 draw IDs may be displayed as 8 digits.
    # Main pagination has omitted these full years; explicitly fetch year-filtered pages.
    for year in (2008, 2009, 2010):
        for page in range(1, 10):
            rows = rows_from_html(fetch((year, page)))
            if not rows:
                if page == 1:
                    raise ValueError(f"year {year}: no rows returned")
                break
            new = 0
            for ident, day, first, second in rows:
                value = (day, first, second)
                if ident in found and found[ident] != value:
                    raise ValueError(f"year {year}: conflicting draw {ident}")
                if ident not in found:
                    found[ident] = value
                    new += 1
            if new == 0:
                break
        else:
            raise ValueError(f"year {year}: pagination cap reached")
    return found

def load_archive(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"draw_id","first","second"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("archive schema mismatch")
        out = {}
        for row in reader:
            ident = row["draw_id"].strip()
            first = tuple(sorted(int(x) for x in row["first"].replace(","," ").split()))
            second = int(row["second"])
            if ident in out:
                raise ValueError(f"duplicate archive id {ident}")
            out[ident] = (first, second)
    return out

def compact_ranges(identifiers):
    """Summarize gaps to distinguish a source coverage gap from pagination failure."""
    if not identifiers:
        return []
    values = sorted(int(x) for x in identifiers)
    ranges = []
    start = previous = values[0]
    for n in values[1:]:
        if n == previous + 1:
            previous = n
        else:
            ranges.append([str(start).zfill(9), str(previous).zfill(9)])
            start = previous = n
    ranges.append([str(start).zfill(9), str(previous).zfill(9)])
    return ranges

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    archive = load_archive(args.archive)
    found = collect()
    missing = sorted(set(archive) - set(found))
    conflicts = sorted(i for i in set(archive) & set(found)
                       if archive[i] != found[i][1:])
    report = {
        "source": SOURCE,
        "source_status": "third_party_not_independently_officially_authenticated",
        "archive_count": len(archive),
        "source_count": len(found),
        "source_first_id": min(found) if found else None,
        "source_last_id": max(found) if found else None,
        "archive_first_id": min(archive) if archive else None,
        "archive_last_id": max(archive) if archive else None,
        "missing_id_ranges": compact_ranges(missing),
        "missing_archive_ids_in_source": missing[:30],
        "missing_count": len(missing),
        "number_conflict_ids": conflicts[:30],
        "number_conflict_count": len(conflicts),
        "archive_sha256": hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
        "official_provenance_verified": False,
        "safe_to_stage": not (missing or conflicts),
        "original_archive_preserved": True,
    }
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if missing or conflicts:
        raise ValueError(f"REJECT: missing={len(missing)}, conflicts={len(conflicts)}; no CSV created")
    out = Path(args.output)
    if out.exists():
        raise ValueError("refuse overwrite")
    buffer = io.StringIO()
    writer = csv.writer(buffer,lineterminator="\n")
    writer.writerow(FIELDS)
    for ident in sorted(archive):
        day, first, second = found[ident]
        writer.writerow((ident, day, " ".join(f"{n:02d}" for n in first), f"{second:02d}"))
    body = buffer.getvalue().encode()
    with out.open("xb") as f:
        f.write(body)
    report["staged_sha256"] = hashlib.sha256(body).hexdigest()
    Path(args.report).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Staged {len(archive)} structurally matched dates, NOT officially authenticated")

if __name__ == "__main__":
    main()
