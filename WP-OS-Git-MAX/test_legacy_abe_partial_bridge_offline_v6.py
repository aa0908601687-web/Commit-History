"""Offline coverage of retired partial neural bridge; never connects it to V6."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import abe_partial_neural_bridge as legacy


def draw_rows():
    return [{"draw_id":str(i),"first":[1,2,3,4,5,6],"second":1} for i in range(20)]

def payload():
    zone={"first":{str(n):1.0 for n in range(1,39)},
          "second":{str(n):1.0 for n in range(1,9)}}
    return {"through_draw":"19","models":{"lstm":zone,"gru":zone}}

def fake_rank(seq, upper, a, b, scalar):
    return {n:float(upper+1-n) for n in range(1,upper+1)}

def fake_normalized(scores):
    total=sum(scores.values())
    return {n:v/total for n,v in scores.items()}


class LegacyPartialBridgeOfflineTests(unittest.TestCase):
    def test_mismatched_cutoff_rejected(self):
        data=payload()
        data["through_draw"]="18"
        with self.assertRaisesRegex(ValueError,"matching cutoff"):
            legacy.bridge(draw_rows(),data)

    def test_missing_neural_model_rejected(self):
        data=payload()
        data["models"].pop("gru")
        with self.assertRaisesRegex(ValueError,"LSTM and GRU"):
            legacy.bridge(draw_rows(),data)

    def test_partial_bridge_preserves_unimplemented_models(self):
        baseline={"through_draw":"19","weights":{"first":{"5_10":0.5},"second":{"5_10":0.5}}}
        with patch.object(legacy,"predict",return_value=baseline),\
             patch.object(legacy,"rank_zone",side_effect=fake_rank),\
             patch.object(legacy,"normalized",side_effect=fake_normalized):
            result=legacy.bridge(draw_rows(),payload())
        self.assertEqual(result["through_draw"],"19")
        self.assertEqual(len(result["first"]),6)
        self.assertTrue(1 <= result["second"] <= 8)
        self.assertFalse(result["full_five_model_fusion_executed"])
        self.assertEqual(result["not_executed"],["chronos_2","t0_gamma"])
        self.assertAlmostEqual(result["effective_weights"]["unallocated_uniform_reserve"],0.2)

    def test_cli_writes_separate_legacy_output(self):
        with tempfile.TemporaryDirectory() as td:
            scores=Path(td)/"partial.json"
            output=Path(td)/"result.json"
            scores.write_text(json.dumps(payload()),encoding="utf-8")
            baseline={"through_draw":"19","weights":{"first":{"5_10":0.5},"second":{"5_10":0.5}}}
            with patch.object(sys,"argv",["bridge","--csv","unused.csv","--trained-scores",str(scores),"--output",str(output)]),\
                 patch.object(legacy.base,"load",return_value=draw_rows()),\
                 patch.object(legacy,"predict",return_value=baseline),\
                 patch.object(legacy,"rank_zone",side_effect=fake_rank),\
                 patch.object(legacy,"normalized",side_effect=fake_normalized):
                legacy.main()
            self.assertEqual(json.loads(output.read_text())["execution_mode"],
                             "abe_lstm_gru_partial_with_uniform_reserve")


if __name__=="__main__":
    unittest.main()
