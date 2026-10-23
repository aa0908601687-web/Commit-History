"""Additional edge-case coverage for WP-OS Git MAX; does not modify prediction code."""
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.util

SOURCE = Path(__file__).with_name("wp_os_git_max.py")
spec = importlib.util.spec_from_file_location("wp_model_edge_cases", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EdgeCoverage(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"draw_id": str(115000001 + i),
             "first": tuple(sorted((1 + ((i + j * 6) % 38)) for j in range(6))),
             "second": i % 8 + 1}
            for i in range(11)
        ]

    def test_score_and_rank_require_ten_draws(self):
        for n in (0, 5, 9):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    m.score([1] * n, 8)
                with self.assertRaises(ValueError):
                    m.rank_draws([self.rows[0]["first"]] * n, 38)

    def test_longer_history_does_not_change_predictions(self):
        self.assertEqual(m.predict(self.rows), m.predict(self.rows[1:]))

    def test_predict_uses_valid_range_and_through_draw(self):
        p = m.predict(self.rows)
        self.assertEqual(p["through_draw"], self.rows[-1]["draw_id"])
        self.assertEqual(len(p["first"]), 6)
        self.assertEqual(len(p["top_first_scores"]), 10)
        self.assertEqual(len(p["second_scores"]), 8)
        self.assertLessEqual(len(p["top_pairs"]), 10)

    def test_backtest_only_counts_observed_draws(self):
        x = m.backtest(self.rows)
        self.assertEqual(x["tested"], 1)
        self.assertEqual(sum(x["first_zone_hits"].values()), 1)
        self.assertIn(x["second_zone_hits"], (0, 1))

    def test_cli_prediction_and_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "draws.csv"
            lines = ["draw_id,first,second"]
            for row in self.rows:
                lines.append(f'{row["draw_id"]},"{" ".join(map(str, row["first"]))}",{row["second"]}')
            file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            for flags in ([], ["--backtest"]):
                with self.subTest(flags=flags):
                    buffer = io.StringIO()
                    with patch.object(sys, "argv", [str(SOURCE), "--csv", str(file)] + flags):
                        with contextlib.redirect_stdout(buffer):
                            m.main()
                    parsed = json.loads(buffer.getvalue())
                    self.assertIn("tested" if flags else "first", parsed)


if __name__ == "__main__":
    unittest.main()
