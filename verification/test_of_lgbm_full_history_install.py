"""Fail-closed normalized source staging regression tests; no actual official data here."""
import csv
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from of_lgbm_full_history_install import stage

class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.source, self.snapshots, self.stage_dir = base/"incoming", base/"snapshots", base/"installed"
        self.source.mkdir()
        self.snapshots.mkdir()
        for game, maxnum, size, official, local, extra in [
            ("power", 38, 6, "power.csv", "power_history_archive.csv", "second"),
            ("lotto", 49, 6, "lotto.csv", "lotto_history_mirror.csv", "special"),
            ("539", 39, 5, "539.csv", "539_history_mirror.csv", None),
        ]:
            for target, dated in [(self.source/official, True), (self.snapshots/local, game != "power")]:
                with target.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["draw_id"] + (["draw_date"] if dated else []) + ["first"] + ([extra] if extra else []))
                    for index in range(20):
                        record = [f"115{index+1:06d}"]
                        if dated:
                            record.append((date(2026,1,1)+timedelta(days=index)).isoformat())
                        record.append("01 02 03 04 05" + (" 06" if size == 6 else ""))
                        if extra:
                            record.append("07" if game == "lotto" else "02")
                        writer.writerow(record)

    def test_stages_exact_match_without_touching_original(self):
        before = (self.snapshots/"power_history_archive.csv").read_bytes()
        data = stage(self.source, self.snapshots, self.stage_dir)
        self.assertEqual(data["datasets"]["power"]["previously_undated_snapshot_rows"], 20)
        self.assertFalse(data["official_provenance_verified"])
        self.assertFalse(data["complete_to_today_verified"])
        self.assertEqual((self.snapshots/"power_history_archive.csv").read_bytes(), before)
        self.assertTrue((self.stage_dir/"STAGING_AUDIT.json").is_file())
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            stage(self.source, self.snapshots, self.stage_dir)

    def test_rejects_number_mismatch_without_install(self):
        p = self.source/"power.csv"
        p.write_text(p.read_text().replace("01 02 03 04 05 06", "01 02 03 04 05 09", 1))
        with self.assertRaisesRegex(ValueError, "draw number mismatch"):
            stage(self.source, self.snapshots, self.stage_dir)
        self.assertFalse(self.stage_dir.exists())

    def test_rejects_missing_source_rows_without_install(self):
        p = self.source/"539.csv"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "does not cover snapshot"):
            stage(self.source, self.snapshots, self.stage_dir)
        self.assertFalse(self.stage_dir.exists())

    def test_rejects_untrusted_missing_date(self):
        p = self.source/"lotto.csv"
        p.write_text(p.read_text().replace("2026-01-02", "", 1))
        with self.assertRaises((ValueError, TypeError)):
            stage(self.source, self.snapshots, self.stage_dir)
        self.assertFalse(self.stage_dir.exists())

    def test_mid_copy_failure_cleans_temporary_stage_and_preserves_sources(self):
        """A failed copy cannot leave partial stage outputs or alter pinned CSVs."""
        import shutil
        from unittest.mock import patch

        original_copy = shutil.copyfile
        before_source = {
            p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()
        }
        before_snapshots = {
            p.name: p.read_bytes() for p in self.snapshots.iterdir() if p.is_file()
        }
        copied = 0

        def fail_on_second_copy(src, dst, *args, **kwargs):
            nonlocal copied
            copied += 1
            if copied == 2:
                raise OSError("injected copy failure")
            return original_copy(src, dst, *args, **kwargs)

        with patch("of_lgbm_full_history_install.shutil.copyfile", side_effect=fail_on_second_copy):
            with self.assertRaisesRegex(OSError, "injected copy failure"):
                stage(self.source, self.snapshots, self.stage_dir)

        self.assertFalse(self.stage_dir.exists())
        self.assertEqual(list(self.stage_dir.parent.glob(".of-lgbm-stage-*")), [])
        self.assertEqual(
            {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()},
            before_source,
        )
        self.assertEqual(
            {p.name: p.read_bytes() for p in self.snapshots.iterdir() if p.is_file()},
            before_snapshots,
        )

if __name__ == "__main__":
    unittest.main()
