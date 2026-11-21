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


    def test_fail_closed_stage_filtered_loader(self):
        import of_lgbm_candidate_registry as registry
        pre = registry.load_candidates(registry.PRE_DRAW)
        tickets = registry.load_candidates(registry.TICKET)
        self.assertEqual(len(pre), 80)
        self.assertEqual(len(tickets), 20)
        self.assertTrue(all(x["stage"] == registry.PRE_DRAW for x in pre))
        self.assertTrue(all(x["stage"] == registry.TICKET for x in tickets))
        with self.assertRaises(ValueError):
            registry.load_candidates("all")

    def test_mutated_metadata_is_rejected(self):
        import copy
        import of_lgbm_candidate_registry as registry
        altered = copy.deepcopy(self.data)
        altered["configurations"][60]["stage"] = registry.PRE_DRAW
        with self.assertRaises(ValueError):
            registry.validate_manifest(altered)
        altered = copy.deepcopy(self.data)
        altered["configurations"][0]["selection"] = "heldout_validation"
        with self.assertRaises(ValueError):
            registry.validate_manifest(altered)
        altered = copy.deepcopy(self.data)
        altered["configurations"][0]["status"] = "active"
        with self.assertRaises(ValueError):
            registry.validate_manifest(altered)
        altered = copy.deepcopy(self.data)
        altered["configurations"][0]["name"] = altered["configurations"][1]["name"]
        with self.assertRaises(ValueError):
            registry.validate_manifest(altered)

if __name__ == "__main__":
    unittest.main()
