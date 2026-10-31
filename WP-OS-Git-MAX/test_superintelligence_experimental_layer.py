import unittest

from superintelligence_experimental_layer import (
    ExperimentalLayerDisabled,
    ExperimentalOrchestrator,
    preprocess_draws,
)


DRAWS = [
    {"draw_id": "1", "first": "01 03 08 17 29 38", "second": "6"},
    {"draw_id": "2", "first": [2, 4, 9, 18, 30, 37], "second": 7},
]


class ExperimentalLayerTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with self.assertRaises(ExperimentalLayerDisabled):
            ExperimentalOrchestrator().run(DRAWS)

    def test_normalizes_and_validates_draws(self):
        rows = preprocess_draws(DRAWS)
        self.assertEqual(rows[0]["first"], [1, 3, 8, 17, 29, 38])
        self.assertEqual(rows[0]["second"], 6)

        with self.assertRaises(ValueError):
            preprocess_draws([{"draw_id": "x", "first": "1 1 2 3 4 5", "second": 1}])

    def test_deterministic_audited_agent_run(self):
        orchestrator = ExperimentalOrchestrator(enabled=True)
        orchestrator.register("count", lambda rows: {"rows": len(rows)})
        first = orchestrator.run(DRAWS)
        second = orchestrator.run(DRAWS)
        self.assertEqual(first, second)
        self.assertFalse(first["asi_claim"])
        self.assertFalse(first["predictive_advantage_claim"])
        self.assertFalse(first["upstream"]["code_imported"])
        self.assertEqual(first["agents"][0]["payload"], {"rows": 2})
        self.assertEqual(len(first["audit_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

