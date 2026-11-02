"""Exercise the retired ABE hybrid CLI output path only; V6 remains independent."""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch
import abe_ultra_hybrid_max as legacy


class LegacyHybridCLITest(unittest.TestCase):
    def test_cli_outputs_mocked_result_without_abe_execution(self):
        fake_rows=[{"draw_id":"115000088","first":[1,2,3,4,5,6],"second":1}]
        fake_result={"through_draw":"115000088","execution_mode":"isolated_test_stub",
                     "full_five_model_fusion_executed":False}
        stream=io.StringIO()
        with patch.object(sys,"argv",["hybrid","--csv","archived.csv"]), \
             patch.object(legacy.base,"load",return_value=fake_rows) as load, \
             patch.object(legacy,"run",return_value=fake_result) as run, \
             contextlib.redirect_stdout(stream):
            legacy.main()
        load.assert_called_once_with("archived.csv")
        run.assert_called_once_with(fake_rows,model_scores_path=None)
        self.assertEqual(json.loads(stream.getvalue()),fake_result)


if __name__=="__main__":
    unittest.main()
