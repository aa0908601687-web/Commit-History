import hashlib
import tempfile
import unittest
from pathlib import Path
from pipeline import sha256_file, train_openfe_lightgbm

class SmokeTests(unittest.TestCase):
    def test_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "sample.csv"
            file.write_bytes(b"draw,number\n1,2\n")
            self.assertEqual(sha256_file(str(file)), hashlib.sha256(file.read_bytes()).hexdigest())

    def test_empty_split_rejected_before_dependency_import(self):
        with self.assertRaises(ValueError):
            train_openfe_lightgbm([], [], [], [])

if __name__ == "__main__":
    unittest.main()
