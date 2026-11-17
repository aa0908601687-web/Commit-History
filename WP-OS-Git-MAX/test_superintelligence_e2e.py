"""End-to-end test: CSV -> production pipeline -> controlled reference bridge."""
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modular_selection_pipeline import run
from superintelligence_bridge import UPSTREAM_COMMIT, analyze
from wp_os_git_max import load


class SuperintelligenceBridgeE2E(unittest.TestCase):
    def test_csv_pipeline_bridge_preserves_predictions(self):
        draws = load(str(HERE / "historical_latest_20.csv"))
        pipeline = run(draws)
        before = pipeline["predictions"]
        report = analyze(pipeline)

        self.assertEqual(pipeline["predictions"], before)
        self.assertEqual(report["through_draw"], draws[-1]["draw_id"])
        self.assertEqual(report["input_pipeline_sha256"], pipeline["sha256"])
        self.assertEqual(report["prediction_weight"], 0.0)
        self.assertFalse(report["selection_mutation_allowed"])
        self.assertFalse(report["upstream_code_executed"])
        self.assertEqual(report["upstream"]["commit"], UPSTREAM_COMMIT)
        self.assertEqual(len(report["sha256"]), 64)

    def test_rejects_cutoff_mismatch(self):
        draws = load(str(HERE / "historical_latest_20.csv"))
        pipeline = run(draws)
        first = next(iter(pipeline["predictions"].values()))
        first["through_draw"] = "invalid"
        with self.assertRaisesRegex(ValueError, "cutoff mismatch"):
            analyze(pipeline)


if __name__ == "__main__":
    unittest.main()
