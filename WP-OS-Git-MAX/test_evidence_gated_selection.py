"""Tests for optional evidence gate (no change to legacy model scoring)."""
import unittest
from unittest.mock import patch
import evidence_gated_selection as gate


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.draws = [{"draw_id": f"{115000001+i:09d}",
                       "first": tuple(sorted(((i * 7 + j * 5) % 38) + 1
                                             for j in range(6))),
                       "second": i % 8 + 1} for i in range(25)]

    def test_fallback_reproducible_valid(self):
        x = gate.fallback(self.draws)
        self.assertEqual(x, gate.fallback(self.draws))
        self.assertEqual(len(set(x["first"])), 6)
        self.assertTrue(all(1 <= n <= 38 for n in x["first"]))
        self.assertTrue(1 <= x["second"] <= 8)

    def test_requires_warmup_and_holdout(self):
        with self.assertRaises(ValueError):
            gate.history_report(self.draws, holdout=6)
        with self.assertRaises(ValueError):
            gate.history_report(self.draws, holdout=0)

    def test_replay_does_not_look_ahead(self):
        cutoff = []
        def fake(seen):
            cutoff.append(seen[-1]["draw_id"])
            return {"first": list(seen[-1]["first"]), "second": seen[-1]["second"]}
        with patch.dict(gate.MODELS, {"mock": fake}, clear=True):
            report = gate.history_report(self.draws, holdout=2)
        self.assertEqual(cutoff, [self.draws[-3]["draw_id"],
                                  self.draws[-2]["draw_id"]])
        self.assertEqual(sum(report["models"]["mock"]["hits_0_to_6"]), 2)

    def test_cli_entry_point(self):
        import contextlib
        import csv
        import io
        import json
        import runpy
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draws.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["draw_id", "first", "second"])
                for d in self.draws:
                    writer.writerow([d["draw_id"],
                                     " ".join(map(str, d["first"])), d["second"]])
            output = io.StringIO()
            with patch.object(sys, "argv", ["evidence_gated_selection.py",
                                            "--csv", str(path), "--holdout", "2"]):
                with contextlib.redirect_stdout(output):
                    runpy.run_module("evidence_gated_selection", run_name="__main__")
            parsed = json.loads(output.getvalue())
            self.assertEqual(parsed["history"]["periods"], 2)
            self.assertEqual(parsed["gate"]["status"], "no_prospective_evidence")

    def test_retrospective_never_claims_verified_advantage(self):
        result = gate.run(self.draws, holdout=2)
        self.assertIsNone(result["gate"]["selected_model"])
        self.assertEqual(result["gate"]["status"], "no_prospective_evidence")
        self.assertEqual(result["candidate"]["method"], "seeded_uniform_fallback")


if __name__ == "__main__":
    unittest.main()
