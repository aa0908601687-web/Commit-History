import unittest
from feature_builder import build_number_features, build_supervised_history


class FeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"first": (1, 2, 3, 4, 5, 6), "second": 1},
            {"first": (2, 3, 4, 5, 6, 7), "second": 2},
            {"first": (3, 4, 5, 6, 7, 8), "second": 3},
        ]

    def test_target_not_in_features(self):
        first = build_number_features(self.rows, 2, "first")
        self.assertEqual(len(first), 38)
        self.assertEqual(first[0]["all_history_hits"], 1)
        self.assertEqual(first[7]["all_history_hits"], 0)

    def test_future_features_use_all_prior(self):
        second = build_number_features(self.rows, len(self.rows), "second")
        self.assertEqual(len(second), 8)
        self.assertEqual(second[2]["last_10_hits"], 1)

    def test_supervised_training_labels(self):
        x, y = build_supervised_history(self.rows, [1, 2], "second")
        self.assertEqual(len(x), 16)
        self.assertEqual(sum(y), 2)
        self.assertEqual(y[1], 1)
        self.assertEqual(y[8 + 2], 1)

    def test_future_label_rejected(self):
        with self.assertRaises(ValueError):
            build_supervised_history(self.rows, [3], "first")


if __name__ == "__main__":
    unittest.main()
