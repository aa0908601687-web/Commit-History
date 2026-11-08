import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_nemotron_ingest import audit_extraction


class NemotronIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.csv_path = root / "verified.csv"
        self.json_path = root / "extracted.json"
        self.draws = []
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["draw_id", "first", "second"])
            for i in range(10):
                numbers = [1+i, 2+i, 3+i, 4+i, 5+i, 6+i]
                draw_id = str(115000001+i)
                writer.writerow([draw_id, " ".join(map(str, numbers)), (i % 8)+1])
                self.draws.append({"draw_id": draw_id, "first": numbers, "second": (i % 8)+1})
        self.save({"source": "image:test-fixture", "draws": self.draws})

    def save(self, payload):
        self.json_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_matching_data(self):
        result = audit_extraction(self.csv_path, self.json_path)
        self.assertEqual(result["checked_draws"], 10)
        self.assertFalse(result["nemotron_model_executed"])

    def test_mismatch_fails_closed(self):
        bad = {"source": "image:test", "draws": [dict(self.draws[0], second=8)]}
        self.save(bad)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit_extraction(self.csv_path, self.json_path)

    def test_unrecognized_draw_fails(self):
        self.save({"source": "image:test", "draws": [{"draw_id": "115999999", "first": [1,2,3,4,5,6], "second": 1}]})
        with self.assertRaisesRegex(ValueError, "absent"):
            audit_extraction(self.csv_path, self.json_path)

    def test_duplicate_fails(self):
        self.save({"source": "image:test", "draws": [self.draws[0], self.draws[0]]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_extraction(self.csv_path, self.json_path)


if __name__ == "__main__":
    unittest.main()
