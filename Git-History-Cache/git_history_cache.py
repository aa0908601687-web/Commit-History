#!/usr/bin/env python3
"""Verified Git cache, reports and optional deterministic Git-seeded lottery picks."""
import argparse
import collections
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile

SCHEMA = 1
POLICY = 'all-reachable-no-renames-root-and-single-parent-v1'
PICK_VERSION = 'wp-git-seeded-sha256-fisher-yates-v1'

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode('utf-8', errors='replace')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()

def valid(obj, repository):
    try:
        if obj['schema_version'] != SCHEMA or obj['policy'] != POLICY or obj['repository'] != repository:
            return False
        if obj['sha256'] != digest({k: v for k, v in obj.items() if k != 'sha256'}):
            return False
        records = obj['commits']
        if not isinstance(records, list) or len({r['sha'] for r in records}) != len(records):
            return False
        if not records or obj.get('head_sha') != records[0]['sha']:
            return False
        for r in records:
            if len(r['sha']) != 40 or any(c not in '0123456789abcdef' for c in r['sha']):
                return False
            for k in ('author', 'email', 'date', 'message'):
                if not isinstance(r[k], str): return False
            datetime.datetime.fromisoformat(r['date'])
            for k in ('additions', 'deletions', 'binary_files'):
                if type(r[k]) is not int or r[k] < 0: return False
            if not isinstance(r['files'], list) or any(not isinstance(f, str) for f in r['files']):
                return False
        return obj['summary'] == summarize(records)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False

def record(repo, sha):
    fields = git(repo, 'show', '-s', '--format=%an%x00%ae%x00%aI%x00%B', sha).split('\0', 3)
    parents = git(repo, 'rev-list', '--parents', '-n', '1', sha).split()[1:]
    # Merges count as commits; omit merge diffs to avoid counting merged changes twice.
    raw = '' if len(parents) > 1 else git(repo, 'show', '--format=', '--numstat', '-z', '--no-renames', sha)
    adds = dels = binary = 0
    files = []
    for line in raw.split('\0'):
        if not line: continue
        a, d, name = line.split('\t', 2)
        files.append(name)
        if a == '-' or d == '-': binary += 1
        else:
            adds += int(a)
            dels += int(d)
    return dict(sha=sha, author=fields[0], email=fields[1], date=fields[2],
                message=fields[3].rstrip('\n'), files=files, additions=adds,
                deletions=dels, binary_files=binary)

def summarize(records):
    authors = collections.defaultdict(lambda: dict(commits=0, additions=0, deletions=0))
    days = collections.defaultdict(lambda: dict(commits=0, additions=0, deletions=0))
    for r in records:
        identity = r['author'] + ' <' + r['email'] + '>'
        day = datetime.datetime.fromisoformat(r['date']).astimezone(datetime.timezone.utc).date().isoformat()
        for row in (authors[identity], days[day]):
            row['commits'] += 1
            row['additions'] += r['additions']
            row['deletions'] += r['deletions']
    for row in authors.values():
        row['percent'] = round(100 * row['commits'] / len(records), 4)
    return dict(commit_count=len(records), by_author=dict(sorted(authors.items())),
                by_day_utc=dict(sorted(days.items())),
                additions=sum(r['additions'] for r in records),
                deletions=sum(r['deletions'] for r in records))

def refresh(repo, path, repository, ref='HEAD'):
    if git(repo, 'rev-parse', '--is-shallow-repository').strip() != 'false':
        raise ValueError('Full Git history required; use fetch-depth: 0')
    head = git(repo, 'rev-parse', '--verify', ref + '^{commit}').strip()
    shas = git(repo, 'rev-list', head).splitlines()
    path = Path(path)
    old = None
    try:
        candidate = json.loads(path.read_text(encoding='utf-8'))
        if valid(candidate, repository): old = candidate
    except (OSError, ValueError):
        pass
    if old and old['head_sha'] == head and [r['sha'] for r in old['commits']] == shas:
        return old, 'hit', 0
    reusable = {r['sha']: r for r in old['commits']} if old else {}
    records = [reusable[sha] if sha in reusable else record(repo, sha) for sha in shas]
    obj = dict(schema_version=SCHEMA, policy=POLICY, repository=repository, head_sha=head,
               generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               commits=records, summary=summarize(records))
    obj['sha256'] = digest(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return obj, 'updated' if old else 'refreshed', sum(s not in reusable for s in shas)

def charts(obj, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, keys in [('contributors', ['commits']), ('code_changes', ['additions', 'deletions'])]:
        rows = list(obj['summary']['by_author'].items())
        height = max(150, 100 + 65 * len(rows))
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<text x="20" y="25">{name}: {html.escape(obj["repository"])}</text>']
        maximum = max([v[k] for _, v in rows for k in keys] + [1])
        for i, (author, vals) in enumerate(rows):
            y = 60 + i * 65
            parts.append(f'<text x="20" y="{y}">{html.escape(author)}</text>')
            for j, key in enumerate(keys):
                color = '#327ba8' if key == 'commits' else '#228855' if key == 'additions' else '#bb4444'
                width = 350 * vals[key] / maximum
                yy = y + j * 23 - 14
                parts.append(f'<rect x="620" y="{yy}" width="{width}" height="18" fill="{color}"/>')
                parts.append(f'<text x="{630+width}" y="{yy+14}">{vals[key]}</text>')
        parts.append('</svg>')
        (out / (name + '.svg')).write_text('\n'.join(parts), encoding='utf-8')

def pick_from_cache(obj, repository, expected_head):
    """Deterministic selection, not a forecast. Reject invalid or stale input."""
    if not valid(obj, repository):
        raise ValueError('Invalid Git history cache')
    if obj['head_sha'] != expected_head:
        raise ValueError('Cache does not match requested commit')
    # Exclude generation timestamps, SVG layout and cache checksum from seed.
    # Include all commit records and the exact data used for contribution charts.
    material = dict(algorithm=PICK_VERSION, repository=repository,
                    head_sha=obj['head_sha'], policy=obj['policy'],
                    commits=sorted(obj['commits'], key=lambda row: row['sha']),
                    summary=obj['summary'])
    seed = digest(material)
    counter = 0

    def below(upper):
        nonlocal counter
        limit = (1 << 256) - ((1 << 256) % upper)
        while True:
            block = hashlib.sha256(bytes.fromhex(seed) + counter.to_bytes(8, 'big')).digest()
            counter += 1
            value = int.from_bytes(block, 'big')
            if value < limit:
                return value % upper

    pool = list(range(1, 39))
    for i in range(6):
        j = i + below(38 - i)
        pool[i], pool[j] = pool[j], pool[i]
    result = dict(mode='git-cache', algorithm=PICK_VERSION, repository=repository,
                  through_commit=obj['head_sha'], source_cache_sha256=obj['sha256'],
                  seed_sha256=seed, first=sorted(pool[:6]), second=below(8) + 1,
                  commit_count=obj['summary']['commit_count'],
                  note='Deterministic Git-derived selection; no draw data or predictive advantage. '
                       'Changing Git data can change the pick; commit timing is not proof of a pre-draw forecast.')
    result['sha256'] = digest(result)
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default='.')
    ap.add_argument('--repository', required=True, help='owner/name cache namespace')
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--report-dir', required=True)
    ap.add_argument('--pick', action='store_true', help='Write git_pick.json from verified current Git cache')
    args = ap.parse_args()
    obj, status, fetched = refresh(args.repo, args.cache, args.repository, args.ref)
    charts(obj, args.report_dir)
    # This report records one lookup; it is not a long-term hit rate.
    total_commits = obj['summary']['commit_count']
    reused_commits = total_commits - fetched
    metrics = dict(lookups=1, hits=int(status == 'hit'),
                   total_commits=total_commits, reused_commits=reused_commits,
                   commit_reuse_percent=round(100 * reused_commits / total_commits, 2) if total_commits else None,
                   misses=int(status != 'hit'), status=status,
                   scope='one Git history cache lookup in this run',
                   new_commits_read=fetched, head_sha=obj['head_sha'])
    (Path(args.report_dir) / 'cache_lookup.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.pick:
        pick = pick_from_cache(obj, args.repository, obj['head_sha'])
        (Path(args.report_dir) / 'git_pick.json').write_text(
            json.dumps(pick, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(pick, ensure_ascii=False, indent=2))
    print(json.dumps(dict(status=status, new_commits_read=fetched, head_sha=obj['head_sha'],
                          summary=obj['summary']), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
