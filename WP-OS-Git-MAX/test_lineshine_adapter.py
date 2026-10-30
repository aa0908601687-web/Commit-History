import unittest
from lineshine_adapter import evaluate

class LineShineReadinessTests(unittest.TestCase):
    def test_missing_secrets_never_claim_connected(self):
        r = evaluate({})
        self.assertFalse(r['configured'])
        self.assertFalse(r['remote_connected'])
        self.assertFalse(r['execution_verified'])
        self.assertEqual(len(r['missing_environment_variables']), 3)

    def test_configured_does_not_mean_connected(self):
        r = evaluate({'LINESHINE_HOST': 'cluster.example.org', 'LINESHINE_USER': 'researcher', 'LINESHINE_SCHEDULER': 'slurm'})
        self.assertTrue(r['configured'])
        self.assertFalse(r['remote_connected'])
        self.assertFalse(r['job_submitted'])

    def test_bad_scheduler_rejected(self):
        r = evaluate({'LINESHINE_HOST': 'cluster.example.org', 'LINESHINE_USER': 'researcher', 'LINESHINE_SCHEDULER': 'unknown'})
        self.assertFalse(r['configured'])
        self.assertTrue(r['errors'])

    def test_localhost_rejected(self):
        r = evaluate({'LINESHINE_HOST': 'localhost', 'LINESHINE_USER': 'researcher', 'LINESHINE_SCHEDULER': 'pbs'})
        self.assertFalse(r['configured'])

if __name__ == '__main__':
    unittest.main()
