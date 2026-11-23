#!/usr/bin/env python3
"""Fresh-download independent evidence verification, not a production write."""
import csv, hashlib, io, json, re, sys, time, zipfile
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_official_zip_date_audit import fetch_live_dates

CANON = Path("verification/of_lgbm_data/power_history_archive.csv")
PRIOR = Path("downloaded-official-stage/DATE_AUDIT.json")
OUT = Path("independent-official-audit.json")
CSV_REF = "84b26e6fbdf740dae94ce5105eb4c463c3dde14ae1f5bd5469b4801f753483de"
CANON_REF = "de92a6019d8efa300f4cbb0e96b7badd9babfea9889b5e7657f0b10a01a80e5b"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def date_value(v):
    v = v.strip().split("T")[0].split(" ")[0]
    v = re.sub("[年月]", "/", v).replace("日", "")
    yr, month, day = map(int, re.split(r"[-/.]", v))
    if yr <= 1911: yr += 1911
    return date(yr, month, day).isoformat()

def read_text(raw):
    for enc in ("utf-8-sig", "cp950", "big5", "utf-16"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: pass
    raise ValueError("unrecognized CSV encoding")

def download(url):
    error = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent":"Mozilla/5.0",
                "Referer":"https://www.taiwanlottery.com/lotto/history/result_download/"})
            with urlopen(req, timeout=45) as response:
                final = response.geturl()
                if not (final.startswith("https://cdn.taiwanlottery.com.tw/") or
                        final.startswith("https://www.taiwanlottery.com/")):
                    raise ValueError("Unexpected non-Taiwan-Lottery redirect: " + final)
                return response.read(20_000_001)
        except Exception as exc:
            error = str(exc)
            if attempt < 2: time.sleep(2 * (attempt+1))
    raise RuntimeError(f"official source download failed: {url}: {error}")

def audit():
    if digest(CANON.read_bytes()) != CANON_REF:
        raise ValueError("canonical archive bytes unexpectedly changed")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    old_sources = prior["source_urls_and_hashes"]
    saved = {int(x["year"]): x for x in old_sources if "year" in x}
    if set(saved) != set(range(2008, 2027)):
        raise ValueError("original audit did not cover all 19 annual ZIPs")
    with CANON.open(encoding="utf-8-sig", newline="") as f:
        canonical = list(csv.DictReader(f))
    reference = {r["draw_id"]: (tuple(sorted(map(int, r["first"].replace(","," ").split()))), int(r["second"])) for r in canonical}
    if len(reference) != len(canonical) or len(reference) != 1964:
        raise ValueError("canonical draw count inconsistent")
    dates = {}
    checked = []
    for year in range(2008, 2027):
        src = saved[year]
        raw = download(src["url"])
        new_hash = digest(raw)
        if new_hash != src["zip_sha256"]:
            raise ValueError(f"{year}: fresh original ZIP differs from pinned artifact SHA-256")
        matched = 0
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for member in z.namelist():
                if not member.lower().endswith(".csv"): continue
                rows = csv.reader(io.StringIO(read_text(z.read(member))))
                headings = next(rows, None)
                if not headings: continue
                headings = [s.strip().lstrip("\ufeff") for s in headings]
                needed = ["遊戲名稱", "期別", "開獎日期"] + ["獎號"+str(i) for i in range(1,7)]
                if not all(h in headings for h in needed): continue
                game_i, draw_i, day_i = (headings.index(h) for h in needed[:3])
                numbers_i = [headings.index(h) for h in needed[3:]]
                extras = [i for i,h in enumerate(headings) if h in ("第二區","特別號")]
                if not extras and headings[-1] == "" and len(headings) == max(numbers_i)+2:
                    extras = [len(headings)-1]
                selected = [r for r in rows if len(r)>game_i and r[game_i].strip()=="威力彩"]
                if not selected: continue
                if len(extras)!=1: raise ValueError(f"ambiguous second-area column in {year}/{member}")
                for r in selected:
                    if len(r) <= max(draw_i,day_i,*numbers_i,extras[0]):
                        raise ValueError(f"truncated official record {year}/{member}")
                    draw_id = r[draw_i].strip().zfill(9)
                    day = date_value(r[day_i])
                    nums = tuple(sorted(int(r[i]) for i in numbers_i))
                    second = int(r[extras[0]])
                    if draw_id in reference:
                        if (nums,second) != reference[draw_id]:
                            raise ValueError(f"{draw_id}: fresh official winning numbers disagree")
                        if draw_id in dates and dates[draw_id]!=day:
                            raise ValueError(f"{draw_id}: conflicting source dates")
                        dates[draw_id]=day
                        matched+=1
        checked.append({"year":year,"zip_sha256":new_hash,"matched_canonical_draws":matched})
        print(json.dumps(checked[-1]), flush=True)
    missing = sorted(set(reference)-set(dates))
    if missing:
        live, src = fetch_live_dates("2026-11", {r["draw_id"]:r for r in canonical})
        for key, day in live.items():
            if key in dates and dates[key]!=day:
                raise ValueError("ZIP and official API date conflict")
            dates[key]=day
        print("Live API supplemental matched",src["matched_draws"],flush=True)
    missing = sorted(set(reference)-set(dates))
    if missing: raise ValueError(f"still missing official dates: {len(missing)}")
    prev = None
    for draw_id in reference:
        day = dates[draw_id]
        if prev and day<=prev: raise ValueError("official dates not chronologically increasing")
        prev=day
    candidate = Path("official-reviewed-Power-dated.csv")
    with candidate.open("x",encoding="utf-8",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["draw_id","draw_date","first","second"])
        for r in canonical: w.writerow([r["draw_id"],dates[r["draw_id"]],r["first"],r["second"]])
    report = {"fresh_annual_zip_downloads":19,"annual_zip_sha256_matched_prior":True,
        "records_independently_compared_with_canonical":len(reference),
        "missing_dates":0,"fresh_zip_checks":checked,
        "dated_csv_sha256":digest(candidate.read_bytes()),
        "prior_candidate_csv_sha256":CSV_REF,
        "matches_prior_candidate_byte_for_byte":digest(candidate.read_bytes())==CSV_REF,
        "official_source_domain":"cdn.taiwanlottery.com.tw",
        "official_source_freshly_retrieved":True,
        "production_import_completed":False,
        "official_provenance_independently_authenticated":False,
        "review_required":"Manual source provenance inspection and release approval required before updating canonical archive",
        "note":"Positive structural and cross-run checks do not alone establish an authority-endorsed certification."}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="fresh_zip_checks"},ensure_ascii=False))
if __name__=="__main__":
    audit()
