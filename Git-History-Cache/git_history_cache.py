#!/usr/bin/env python3
"""Verified local Git history cache and SVG reports; no lottery input changes."""
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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default='.')
    ap.add_argument('--repository', required=True, help='owner/name cache namespace')
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--report-dir', required=True)
    args = ap.parse_args()
    obj, status, fetched = refresh(args.repo, args.cache, args.repository, args.ref)
    charts(obj, args.report_dir)
    print(json.dumps(dict(status=status, new_commits_read=fetched, head_sha=obj['head_sha'],
                          summary=obj['summary']), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
