"""Offline tests for the V6 optional TabPrep integration contract."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v6_ai_qaoa_e2e import run


class TabPrepContractTests(unittest.TestCase):
    def test_mutual_exclusion_before_execution(self):
        with self.assertRaisesRegex(ValueError, "isolated experimental"):
            run("missing.csv", openfe_three_views=True, tabprep_three_views=True)

    def test_pytorch_fusion_not_silently_mixed(self):
        with self.assertRaisesRegex(ValueError, "isolated experimental"):
            run("missing.csv", pytorch_three_views=True, tabprep_three_views=True)

    def test_optional_mode_is_explicit(self):
        from v6_tabprep_bridge import score_three_history_views_tabprep
        with self.assertRaisesRegex(ValueError, "at least 12"):
            score_three_history_views_tabprep([])

    def test_missing_dependency_fails_closed(self):
        # Does not require installing the heavyweight upstream package in CI.
        import builtins
        original = builtins.__import__
        def deny(name, *args, **kwargs):
            if name == "tabprep":
                raise ImportError("tabprep absent")
            return original(name, *args, **kwargs)
        rows = [("115000001", [1,2,3,4,5,6], 1)] * 12
        with patch("builtins.__import__", side_effect=deny):
            with self.assertRaisesRegex(RuntimeError, "optional and not installed"):
                from v6_tabprep_bridge import score_three_history_views_tabprep
                score_three_history_views_tabprep(rows)


if __name__ == "__main__":
    unittest.main()
