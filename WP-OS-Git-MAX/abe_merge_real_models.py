#!/usr/bin/env python3
"""Merge actual trained LSTM/GRU and pretrained Chronos-2/t0 scores only."""
import argparse
import json
from pathlib import Path
from abe_score_adapters import validate_scores

def merge(recurrent, foundation):
    if recurrent.get("through_draw") != foundation.get("through_draw"):
        raise ValueError("Mismatched model cutoff: cannot fuse")
    if recurrent.get("mode") != "trained_local_lstm_gru_only" or set(recurrent.get("models",{})) != {"lstm","gru"}:
        raise ValueError("Expected trained recurrent output")
    if foundation.get("execution_mode") != "real_pretrained_inference" or set(foundation.get("models",{})) != {"chronos_2","t0_gamma"}:
        raise ValueError("Expected real pretrained foundation output")
    result = {"through_draw": recurrent["through_draw"],
              "models": {**recurrent["models"], **foundation["models"]},
              "input_provenance": {"lstm_gru": recurrent["mode"],
                                   "foundation": foundation["execution_mode"]}}
    validate_scores(result, result["through_draw"])
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--recurrent",required=True)
    p.add_argument("--foundation",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=merge(json.loads(Path(a.recurrent).read_text(encoding="utf-8")),
                 json.loads(Path(a.foundation).read_text(encoding="utf-8")))
    Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
