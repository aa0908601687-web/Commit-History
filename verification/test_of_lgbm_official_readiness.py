import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('official_gate', Path(__file__).with_name('of_lgbm_official_readiness.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class OfficialReadinessTests(unittest.TestCase):
    def test_missing_evidence_is_blocked(self):
        self.assertFalse(module.audit({})['production_official_data_ready'])

    def test_unverified_rows_are_blocked(self):
        m = {'record_by_record_officially_authenticated': False,
             'datasets': {k: {'official_row_verification':'not_completed', 'sha256':'abc'} for k in module.GAMES}}
        self.assertFalse(module.audit(m)['production_official_data_ready'])

    def test_complete_manifest_only_clears_metadata_gate(self):
        m = {'record_by_record_officially_authenticated': True,
             'datasets': {k: {'official_row_verification':'completed', 'sha256':'abc', 'date_status':'complete'} for k in module.GAMES}}
        r = module.audit(m)
        self.assertTrue(r['production_official_data_ready'])
        self.assertFalse(r['official_authentication_performed_by_this_check'])

if __name__ == '__main__':
    unittest.main()
