"""Additional V6 OpenFE/cache contract and corruption regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys

import v6_openfe_bridge as openfe
import v6_of3q_cache_v3 as cache


class OpenFEContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            (f"1150000{i:02d}", [1, 2, 3, 4, 5, (i % 33) + 6], (i % 8) + 1)
            for i in range(1, 25)
        ]

    def test_base_row_uses_only_completed_history(self):
        history = self.rows[:10]
        row = openfe._base_row(history, 1, 10)
        self.assertEqual(row["freq_5"], 5.0)
        self.assertEqual(row["freq_10"], 10.0)
        self.assertEqual(row["freq_view"], 1.0)
        self.assertEqual(row["gap_view"], 1.0)
        self.assertEqual(row["view_is_10"], 1.0)
        self.assertEqual(row["view_is_full"], 0.0)
        self.assertEqual(row["view_rows"], 10.0)

    def test_missing_number_gap_is_bounded_by_view(self):
        row = openfe._base_row(self.rows[:10], 38, 10)
        self.assertEqual(row["gap_view"], 11.0)
        self.assertEqual(row["freq_view"], 0.0)
        self.assertEqual(row["low_zone"], 0.0)

    def test_invalid_history_rejected_before_importing_openfe(self):
        with self.assertRaises(ValueError):
            openfe.score_first_zone(self.rows[:11])
        with self.assertRaises(ValueError):
            openfe.score_three_history_views(self.rows[:19])

    def test_score_first_zone_mapping_with_stubbed_fit(self):
        with patch.dict(sys.modules, {"pandas": SimpleNamespace(DataFrame=lambda values: values)}), \
             patch.object(openfe, "_training_frame", return_value=("x", "y")), \
             patch.object(openfe, "_base_row", return_value={"number": 1.0}), \
             patch.object(openfe, "_fit_and_score", return_value=([0.25] * 38, 3, 2)):
            values, meta = openfe.score_first_zone(self.rows)
        self.assertEqual(len(values), 38)
        self.assertEqual(values[38], 0.25)
        self.assertEqual(meta["used_features"], 2)

    def test_three_view_fusion_and_metadata_with_stubbed_fit(self):
        probabilities = [0.2] * 38 + [0.4] * 38 + [0.6] * 38
        with patch.dict(sys.modules, {"pandas": SimpleNamespace(DataFrame=lambda values: values)}), \
             patch.object(openfe, "_training_frame", return_value=([1] * (38 * 3), "y")), \
             patch.object(openfe, "_base_row", return_value={"number": 1.0}), \
             patch.object(openfe, "_fit_and_score", return_value=(probabilities, 7, 4)):
            values, meta = openfe.score_three_history_views(self.rows)
        self.assertAlmostEqual(values[1], 0.4)
        self.assertEqual(len(meta["per_view_scores"]["latest_20"]), 38)
        self.assertEqual(meta["training_anchor_count"], 1)
        self.assertEqual(meta["used_features"], 4)


class CacheIntegrityTests(unittest.TestCase):
    def test_digest_is_independent_of_dictionary_order(self):
        self.assertEqual(cache.digest({"b": 2, "a": 1}), cache.digest({"a": 1, "b": 2}))

    def test_persisted_record_round_trip_and_key_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "entry.json"
            original = cache._write(path, {"schema_version": 3, "layer": "data", "rows": [1, 2]})
            self.assertEqual(cache._read(path, {"layer": "data"}), original)
            self.assertIsNone(cache._read(path, {"layer": "features"}))

    def test_missing_malformed_and_tampered_cache_is_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entry.json"
            self.assertIsNone(cache._read(path, {}))
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(cache._read(path, {}))
            cache._write(path, {"layer": "data", "rows": [1, 2]})
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["rows"].append(3)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(cache._read(path, {"layer": "data"}))

    def test_feature_cache_hit_converts_number_keys_back_to_int(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            key = {"schema_version": cache.SCHEMA_VERSION, "layer": "features",
                   "data_sha256": "source", "feature_version": cache.FEATURE_VERSION,
                   "seed": cache.SEED}
            cache._write(path, {**key, "scores": {1: 0.5}, "metadata": {"engine": "stub"}})
            with patch.object(cache, "score_three_history_views", side_effect=AssertionError("unexpected fit")):
                values, meta, status, sha = cache.feature_layer([], "source", path)
            self.assertEqual(values, {1: 0.5})
            self.assertEqual(status, "hit")
            self.assertEqual(meta["engine"], "stub")
            self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main()
