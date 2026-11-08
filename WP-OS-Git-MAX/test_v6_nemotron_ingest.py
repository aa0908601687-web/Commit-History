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


    def test_malformed_payloads_fail_closed(self):
        valid = self.draws[0]
        cases = [
            ([], "nonempty"), ({}, "nonempty"),
            ({"draws": []}, "nonempty"),
            ({"source": "s", "draws": "not-list"}, "nonempty"),
            ({"source": "", "draws": [valid]}, "source"),
            ({"source": 7, "draws": [valid]}, "source"),
            ({"source": "s", "draws": [None]}, "exactly"),
            ({"source": "s", "draws": [dict(valid, extra=1)]}, "exactly"),
            ({"source": "s", "draws": [dict(valid, draw_id=5)]}, "draw_id"),
            ({"source": "s", "draws": [dict(valid, draw_id="1")]}, "draw_id"),
            ({"source": "s", "draws": [dict(valid, draw_id="abcdefghi")]}, "draw_id"),
            ({"source": "s", "draws": [dict(valid, first="1 2 3 4 5 6")]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, first=[1,2])]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, first=[1,1,2,3,4,5])]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, first=[0,2,3,4,5,6])]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, first=[True,2,3,4,5,6])]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, second=True)]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, second=9)]}, "lottery"),
            ({"source": "s", "draws": [dict(valid, second=0)]}, "lottery"),
        ]
        for payload, msg in cases:
            with self.subTest(payload=payload):
                self.save(payload)
                with self.assertRaisesRegex(ValueError, msg):
                    audit_extraction(self.csv_path, self.json_path)

    def test_main_passes_verified_audit_to_v6_without_changing_scores(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        import v6_nemotron_ingest
        self.save({"source": "fixture", "draws": self.draws})
        args = ["prog", "--csv", str(self.csv_path),
                "--nemotron-extraction", str(self.json_path),
                "--openfe-three-views", "--pytorch-three-views",
                "--pytorch-weight", "0.3", "--require-cudaq",
                "--ising-qubo-check"]
        output = io.StringIO()
        with patch.object(sys, "argv", args), patch.object(
            v6_nemotron_ingest, "run", return_value={"candidate_pool": list(range(1,9))}
        ) as mocked, redirect_stdout(output):
            v6_nemotron_ingest.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["nemotron_audit"]["status"], "matched_canonical_csv")
        mocked.assert_called_once_with(
            str(self.csv_path), openfe_three_views=True,
            pytorch_three_views=True, pytorch_weight=0.3,
            use_cudaq=True, require_cudaq=True, ising_qubo_check=True)
        output = io.StringIO()
        with patch.object(sys, "argv", ["prog", "--csv", str(self.csv_path),
             "--nemotron-extraction", str(self.json_path), "--cudaq"]), patch.object(
             v6_nemotron_ingest, "run", return_value={}
        ) as mocked, redirect_stdout(output):
            v6_nemotron_ingest.main()
        self.assertTrue(mocked.call_args.kwargs["use_cudaq"])


if __name__ == "__main__":
    unittest.main()
