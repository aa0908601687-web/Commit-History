"""Dependency-free unit tests for the independent tabular data pipeline."""
import csv
import tempfile
import unittest
from pathlib import Path
from data_pipeline import (validate_csv, chronological_split,
                           evaluate_classification, evaluate_regression,
                           sha256_file, write_audit)

class DataPipelineTests(unittest.TestCase):
    def test_csv_split_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"data.csv"
            p.write_text("time,x,target\n2026-01-01,1,A\n2026-01-02,2,B\n2026-01-03,3,A\n2026-01-04,4,B\n", encoding="utf-8")
            report = validate_csv(p, "target", "time")
            self.assertEqual(report["row_count"], 4)
            train, test = chronological_split(report["rows"], 0.75)
            self.assertEqual((len(train), len(test)), (3, 1))
            self.assertEqual(report["sha256"], sha256_file(p))
            audit = write_audit(Path(d)/"audit.json", p, {"train":3,"test":1})
            self.assertEqual(len(audit["report_sha256"]), 64)
    def test_validation_rejects_bad_data(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.csv"
            p.write_text("time,target\n2026-01-02,A\n2026-01-01,B\n2026-01-03,C\n")
            with self.assertRaises(ValueError):
                validate_csv(p, "target", "time")
    def test_metrics(self):
        self.assertEqual(evaluate_classification([1,0],[1,1])["accuracy"], .5)
        self.assertAlmostEqual(evaluate_regression([1,3],[2,2])["mae"],1)
if __name__ == "__main__":
    unittest.main()
