"""Static registry checks only; this does not train models or verify E2E."""
import json
import unittest
from pathlib import Path

MANIFEST = Path(__file__).with_name("of_lgbm_100_candidate_configs.json")
GROUPS = {"frequency_time", "cooccurrence", "repeat_streak_omission", "ticket_structure", "openfe_interactions"}

class TestStandaloneCandidateRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.items = cls.data["configurations"]

    def test_100_distinct_configurations(self):
        self.assertEqual(len(self.items), 100)
        self.assertEqual([x["id"] for x in self.items], [f"{i:03d}" for i in range(1, 101)])
        self.assertEqual(len(set(x["name"] for x in self.items)), 100)

    def test_groups_and_stage_isolation(self):
        self.assertEqual({x["group"] for x in self.items}, GROUPS)
        for group in GROUPS:
            self.assertEqual(sum(x["group"] == group for x in self.items), 20)
        for row in self.items:
            stage = "candidate_ticket_scoring" if row["group"] == "ticket_structure" else "pre_draw_feature_proposal"
            self.assertEqual(row["stage"], stage)

    def test_no_unverified_activation_or_fixed_weights(self):
        self.assertIs(self.data["fixed_weights"], False)
        for row in self.items:
            self.assertEqual(row["selection"], "train_fold_only")
            self.assertEqual(row["status"], "proposed_not_integrated")

if __name__ == "__main__":
    unittest.main()
