"""Additional integrity tests for V6's archive parser and EXACT research selection."""
import csv
import tempfile
import unittest
from pathlib import Path

import v6_ai_exact_e2e as v6


class V6EXACTBoundaryTests(unittest.TestCase):
    def make_archive(self, count=10):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "draws.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["draw_id", "first", "second"])
            for i in range(count):
                writer.writerow([f"{115000001+i:09d}", "01 02 03 04 05 06", "01"])
        return tmp, path

    def test_exactly_ten_draws_is_accepted(self):
        tmp, path = self.make_archive()
        with tmp:
            result = v6.run(path)
            self.assertEqual(result["views"]["complete_archive"]["rows"], 10)
            self.assertEqual(result["views"]["latest_5"]["rows"], 5)
            self.assertEqual(len(result["first_zone"]), 6)

    def test_empty_draw_rows_rejected(self):
        tmp, path = self.make_archive(0)
        with tmp, self.assertRaises(ValueError):
            v6.validate(path)

    def test_duplicate_last_draw_is_rejected(self):
        tmp, path = self.make_archive()
        with tmp:
            content = path.read_text(encoding="utf-8")
            rows = content.splitlines()
            rows[-1] = rows[-2]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad draw"):
                v6.validate(path)

    def test_nonmonotonic_draw_order_is_rejected(self):
        tmp, path = self.make_archive()
        with tmp:
            rows = path.read_text(encoding="utf-8").splitlines()
            rows[2], rows[3] = rows[3], rows[2]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "draw order"):
                v6.validate(path)


if __name__ == "__main__":
    unittest.main()
