import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from abe_score_adapters import validate_scores, fuse, NAMES


class AdapterTests(unittest.TestCase):
    def fixture(self):
        zones = {z: {str(i): 1 for i in range(1, n + 1)}
                 for z, n in (("first", 38), ("second", 8))}
        return {"through_draw": "115000086",
                "models": {name: {"first": dict(zones["first"]),
                                  "second": dict(zones["second"])} for name in NAMES}}

    def test_valid_fusion(self):
        m = validate_scores(self.fixture(), "115000086")
        abe = {zone: {n: 1 / count for n in range(1, count + 1)}
               for zone, count in (("first", 38), ("second", 8))}
        result = fuse(abe, m, {"abe_bayesian": .6, **{name: .1 for name in NAMES}})
        self.assertEqual(result["first"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(result["second"], 1)

    def test_reject_stale(self):
        with self.assertRaises(ValueError):
            validate_scores(self.fixture(), "115000087")

    def test_reject_missing_model(self):
        f = self.fixture()
        del f["models"]["t0_gamma"]
        with self.assertRaises(ValueError):
            validate_scores(f, "115000086")

    def test_reject_incomplete_and_negative(self):
        f = self.fixture()
        del f["models"]["lstm"]["second"]["8"]
        with self.assertRaises(ValueError):
            validate_scores(f, "115000086")
        f = self.fixture()
        f["models"]["lstm"]["first"]["1"] = -1
        with self.assertRaises(ValueError):
            validate_scores(f, "115000086")


if __name__ == "__main__":
    unittest.main()
