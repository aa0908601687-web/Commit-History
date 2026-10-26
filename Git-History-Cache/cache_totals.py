#!/usr/bin/env python3
"""Persist one validated lookup per workflow run in a dedicated Git branch."""
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

BRANCH = 'wp-cache-metrics'
PATH = 'cache-metrics/git-history.json'

def incorporate(state, run_id, outcome):
    if state.get('version') != 1 or not isinstance(state.get('runs'), dict):
        raise ValueError('Invalid metrics state')
    if outcome not in ('hit', 'updated', 'refreshed') or not run_id.isdecimal():
        raise ValueError('Invalid lookup outcome or run id')
    runs = state['runs']
    if any(not str(k).isdecimal() or v not in ('hit', 'updated', 'refreshed') for k, v in runs.items()):
        raise ValueError('Invalid stored run')
    if run_id in runs and runs[run_id] != outcome:
        # A rerun can see a different cache: retain the first recorded result.
        return state
    runs.setdefault(run_id, outcome)
    return state

def report(state):
    runs = state['runs']
    hits = sum(value == 'hit' for value in runs.values())
    total = len(runs)
    return dict(lookups=total, hits=hits, misses=total-hits,
                hit_rate_percent=round(100*hits/total, 2) if total else None,
                scope='Git history cache; one first recorded lookup per main-branch workflow run',
                excludes='Runs before metrics were enabled and runs that could not save metrics')

def api(method, path, payload=None):
    url = os.environ['GITHUB_API_URL'] + '/repos/' + os.environ['GITHUB_REPOSITORY'] + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        'Authorization': 'Bearer ' + os.environ['GH_METRICS_TOKEN'],
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def main():
    lookup = json.loads(Path('git-history-report/cache_lookup.json').read_text())
    if lookup['lookups'] != 1 or lookup['hits'] != int(lookup['status'] == 'hit') or lookup['misses'] != 1-lookup['hits']:
        raise ValueError('Invalid lookup report')
    run_id = os.environ['GITHUB_RUN_ID']
    file_path = '/contents/' + PATH
    try:
        api('GET', '/git/ref/heads/' + BRANCH)
    except urllib.error.HTTPError as error:
        if error.code != 404: raise
        try:
            api('POST', '/git/refs', {'ref': 'refs/heads/' + BRANCH, 'sha': os.environ['GITHUB_SHA']})
        except urllib.error.HTTPError as race:
            if race.code != 422: raise
    for attempt in range(10):
        try:
            current = api('GET', file_path + '?ref=' + urllib.parse.quote(BRANCH))
            state = json.loads(base64.b64decode(current['content']))
            sha = current['sha']
        except urllib.error.HTTPError as error:
            if error.code != 404: raise
            state, sha = {'version': 1, 'runs': {}}, None
        already_recorded = run_id in state.get('runs', {})
        incorporate(state, run_id, lookup['status'])
        if not already_recorded:
            payload = {'message': 'Record Git history cache lookup', 'branch': BRANCH,
                       'content': base64.b64encode(json.dumps(state, sort_keys=True, indent=2).encode()).decode()}
            if sha: payload['sha'] = sha
            try:
                api('PUT', file_path, payload)
            except urllib.error.HTTPError as error:
                if error.code not in (409, 422) or attempt == 9: raise
                time.sleep(min(attempt + 1, 5))
                continue
        totals = report(state)
        Path('git-history-report/cache_totals.json').write_text(json.dumps(totals, indent=2) + '\n')
        print(json.dumps(totals, indent=2))
        return
    raise RuntimeError('Unable to save metrics after retries')

if __name__ == '__main__':
    main()
