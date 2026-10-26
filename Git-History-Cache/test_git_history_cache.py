import json
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import git_history_cache as cache

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.run_git('init', '-q')
        self.run_git('config', 'user.name', 'Test & author')
        self.run_git('config', 'user.email', 'test@example.invalid')
        self.path = self.root / 'cache.json'
        self.commit('one\ntwo\n')

    def run_git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args])

    def commit(self, contents):
        (self.repo / 'a.txt').write_text(contents)
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'test commit')

    def refresh(self):
        return cache.refresh(self.repo, self.path, 'owner/repo')

    def test_hit_and_incremental_and_rewind(self):
        obj, status, n = self.refresh()
        self.assertEqual((status, n), ('refreshed', 1))
        self.assertEqual(obj['summary']['additions'], 2)
        original = self.path.read_bytes()
        with patch.object(cache, 'record', side_effect=AssertionError('must reuse')):
            self.assertEqual(self.refresh()[1:], ('hit', 0))
        self.assertEqual(original, self.path.read_bytes())
        self.commit('one\nthree\nfour\n')
        with patch.object(cache, 'record', wraps=cache.record) as reader:
            obj, status, n = self.refresh()
            self.assertEqual(reader.call_count, 1)
        self.assertEqual((status, n), ('updated', 1))
        self.assertEqual((obj['summary']['commit_count'], obj['summary']['additions'], obj['summary']['deletions']), (2, 4, 1))
        self.run_git('reset', '--hard', 'HEAD~1')
        self.assertEqual(self.refresh()[0]['summary']['commit_count'], 1)

    def test_corruption_schema_and_namespace(self):
        for kind in ('checksum', 'schema', 'repository', 'summary', 'json'):
            obj = self.refresh()[0]
            if kind == 'json': self.path.write_text('{')
            else:
                if kind == 'checksum': obj['commits'][0]['message'] = 'tamper'
                if kind == 'schema': obj['schema_version'] = 999
                if kind == 'repository': obj['repository'] = 'different/repo'
                if kind == 'summary': obj['summary']['commit_count'] = 99
                if kind != 'checksum': obj['sha256'] = cache.digest({k:v for k,v in obj.items() if k != 'sha256'})
                self.path.write_text(json.dumps(obj))
            self.assertEqual(self.refresh()[1:], ('refreshed', 1))

    def test_binary_files_and_svg_escape(self):
        (self.repo / 'binary').write_bytes(b'\x00\xff')
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'binary')
        obj = self.refresh()[0]
        self.assertEqual(obj['commits'][0]['binary_files'], 1)
        cache.charts(obj, self.root / 'reports')
        self.assertIn('Test &amp; author', (self.root / 'reports/contributors.svg').read_text())

    def test_shallow_rejected(self):
        dest = self.root / 'shallow'
        subprocess.check_call(['git', 'clone', '-q', '--depth', '1', self.repo.as_uri(), str(dest)])
        with self.assertRaisesRegex(ValueError, 'Full Git history'):
            cache.refresh(dest, self.path, 'owner/repo')

    def test_merge_counts_without_duplicate_changes(self):
        branch = self.run_git('branch', '--show-current').decode().strip()
        self.run_git('checkout', '-qb', 'side')
        (self.repo / 'side.txt').write_text('side\n')
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'side')
        self.run_git('checkout', '-q', branch)
        self.commit('main\n')
        self.run_git('merge', '--no-ff', '-qm', 'merge', 'side')
        obj = self.refresh()[0]
        self.assertEqual(obj['summary']['commit_count'], 4)
        self.assertEqual(obj['commits'][0]['files'], [])
        self.assertEqual((obj['summary']['additions'], obj['summary']['deletions']), (4, 2))

    def test_cli(self):
        args = ['--repo', str(self.repo), '--repository', 'owner/repo', '--cache', str(self.path), '--report-dir', str(self.root / 'reports')]
        result = subprocess.check_output(['python3', str(Path(cache.__file__)), *args], text=True)
        self.assertEqual(json.loads(result)['status'], 'refreshed')
        first_metrics = json.loads((self.root / 'reports/cache_lookup.json').read_text())
        self.assertEqual((first_metrics['misses'], first_metrics['reused_commits'], first_metrics['commit_reuse_percent']), (1, 0, 0))
        result = subprocess.check_output(['python3', str(Path(cache.__file__)), *args], text=True)
        self.assertEqual(json.loads(result)['status'], 'hit')
        metrics = json.loads((self.root / 'reports/cache_lookup.json').read_text())
        self.assertEqual((metrics['lookups'], metrics['hits'], metrics['misses']), (1, 1, 0))
        self.assertEqual((metrics['total_commits'], metrics['reused_commits'], metrics['commit_reuse_percent']), (1, 1, 100))
        self.commit('another\n')
        subprocess.check_output(['python3', str(Path(cache.__file__)), *args], text=True)
        metrics = json.loads((self.root / 'reports/cache_lookup.json').read_text())
        self.assertEqual((metrics['status'], metrics['total_commits'], metrics['reused_commits'], metrics['commit_reuse_percent']), ('updated', 2, 1, 50))

    def test_pick_stable_and_valid(self):
        obj = self.refresh()[0]
        first = cache.pick_from_cache(obj, 'owner/repo', obj['head_sha'])
        for _ in range(20):
            self.assertEqual(first, cache.pick_from_cache(obj, 'owner/repo', obj['head_sha']))
        self.assertEqual(len(set(first['first'])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in first['first']))
        self.assertTrue(1 <= first['second'] <= 8)
        changed = copy.deepcopy(obj)
        changed['generated_utc'] = '2030-01-01T00:00:00+00:00'
        changed['sha256'] = cache.digest({k:v for k,v in changed.items() if k != 'sha256'})
        self.assertEqual(first['seed_sha256'], cache.pick_from_cache(changed, 'owner/repo', obj['head_sha'])['seed_sha256'])
        self.commit('new\n')
        updated = self.refresh()[0]
        self.assertNotEqual(first['seed_sha256'], cache.pick_from_cache(updated, 'owner/repo', updated['head_sha'])['seed_sha256'])

    def test_pick_rejects_tampered_wrong_repo_and_stale(self):
        obj = self.refresh()[0]
        with self.assertRaisesRegex(ValueError, 'requested commit'):
            cache.pick_from_cache(obj, 'owner/repo', '0' * 40)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            cache.pick_from_cache(obj, 'other/repo', obj['head_sha'])
        bad = copy.deepcopy(obj)
        bad['summary']['additions'] += 1
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            cache.pick_from_cache(bad, 'owner/repo', obj['head_sha'])
        bad['sha256'] = cache.digest({k:v for k,v in bad.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            cache.pick_from_cache(bad, 'owner/repo', obj['head_sha'])

    def test_pick_cli_artifact(self):
        out = self.root / 'reports'
        args = ['python3', str(Path(cache.__file__)), '--repo', str(self.repo),
                '--repository', 'owner/repo', '--cache', str(self.path),
                '--report-dir', str(out), '--pick']
        subprocess.check_output(args)
        first = json.loads((out / 'git_pick.json').read_text())
        subprocess.check_output(args)
        self.assertEqual(first, json.loads((out / 'git_pick.json').read_text()))
        self.assertEqual(first['source_cache_sha256'], json.loads(self.path.read_text())['sha256'])

if __name__ == '__main__':
    unittest.main()
