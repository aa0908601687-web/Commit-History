"""Coverage for the verified live-merge CSV export used by the full V6 E2E."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v6_live_api_e2e_probe as live


class TestVerifiedMergedOutput(unittest.TestCase):
    def test_probe_writes_verified_csv_for_full_v6(self):
        stored = [(str(115000100 + i), [1, 2, 3, 4, 5, 6], 1) for i in range(20)]
        remote = [stored[-1], ("115000120", [2, 3, 4, 5, 6, 7], 2)]

        def fetcher(month):
            return remote if month == "2026-11" else []

        def bridge(path):
            data = Path(path).read_bytes()
            with Path(path).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            return {"cutoff": rows[-1]["draw_id"],
                    "source_archive_sha256": hashlib.sha256(data).hexdigest()}

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "live-merged.csv"
            with patch.object(live.official, "existing", return_value=stored):
                result = live.probe(month="2026-11", archive_path=Path(folder) / "archive.csv",
                                    fetcher=fetcher, bridge=bridge, merged_output=destination)
            self.assertTrue(destination.is_file())
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(),
                             result["verified_merged_sha256"])
            self.assertEqual(result["merged_cutoff"], "115000120")
            self.assertEqual(result["new_draw_count"], 1)
            with destination.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 21)
            self.assertEqual(rows[-1]["draw_id"], "115000120")


if __name__ == "__main__":
    unittest.main()
