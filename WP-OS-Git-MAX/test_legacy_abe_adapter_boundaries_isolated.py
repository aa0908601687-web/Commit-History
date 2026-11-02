"""Isolated legacy ABE adapter input-boundary tests.

This covers legacy validation only; never imports or integrates ABE into V6.
"""
import copy
import unittest
import abe_score_adapters as adapter


def valid_payload():
    return {
        "through_draw": "115000088",
        "models": {
            name: {
                "first": {str(i): 1.0 for i in range(1, 39)},
                "second": {str(i): 1.0 for i in range(1, 9)},
            }
            for name in adapter.NAMES
        },
    }


class LegacyAdapterValidationTests(unittest.TestCase):
    def test_rejects_invalid_zone_shape(self):
        payload = valid_payload()
        payload["models"]["lstm"] = {"first": payload["models"]["lstm"]["first"]}
        with self.assertRaisesRegex(ValueError, "Invalid zones"):
            adapter.validate_scores(payload, "115000088")

    def test_rejects_non_dictionary_zone_scores(self):
        payload = valid_payload()
        payload["models"]["gru"]["first"] = [1.0] * 38
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            adapter.validate_scores(payload, "115000088")

    def test_rejects_zero_sum_scores(self):
        payload = valid_payload()
        payload["models"]["chronos_2"]["second"] = {str(i): 0.0 for i in range(1, 9)}
        with self.assertRaisesRegex(ValueError, "positive finite sum"):
            adapter.validate_scores(payload, "115000088")

    def test_rejects_nonfinite_total_even_with_finite_values(self):
        payload = valid_payload()
        payload["models"]["t0_gamma"]["first"] = {str(i): 1e308 for i in range(1, 39)}
        with self.assertRaisesRegex(ValueError, "positive finite sum"):
            adapter.validate_scores(payload, "115000088")

    def test_rejects_invalid_fusion_weight_keys(self):
        with self.assertRaisesRegex(ValueError, "Invalid weight keys"):
            adapter.fuse({}, {}, {"abe_bayesian": 1.0})

    def test_rejects_nonfinite_fusion_weight(self):
        weights = {"abe_bayesian": float("nan"), **{n: 0.0 for n in adapter.NAMES}}
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            adapter.fuse({}, {}, weights)

    def test_rejects_fusion_weight_sum_mismatch(self):
        weights = {"abe_bayesian": 0.8, **{n: 0.0 for n in adapter.NAMES}}
        with self.assertRaisesRegex(ValueError, "sum to one"):
            adapter.fuse({}, {}, weights)


if __name__ == "__main__":
    unittest.main()
