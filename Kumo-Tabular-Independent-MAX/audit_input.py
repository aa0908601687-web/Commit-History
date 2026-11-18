"""CLI for validating and auditing real tabular input before Kumo inference.

This tool does not claim to run the Kumo model. It creates a reproducible
validated manifest and chronological split inventory for downstream inference.
"""
import argparse
import json
from pathlib import Path
from data_pipeline import validate_csv, chronological_split, write_audit

def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate and audit Kumo input CSV")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--time-column")
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("data_audit.json"))
    args = parser.parse_args(argv)
    report = validate_csv(args.csv, args.target, args.time_column)
    train, test = chronological_split(report["rows"], args.split)
    manifest = {
        "input_rows": report["row_count"],
        "train_rows": len(train),
        "test_rows": len(test),
        "columns": report["fields"],
        "target": args.target,
        "time_column": args.time_column,
        "split_fraction": args.split,
        "note": "Input audit only; Kumo model inference not performed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = write_audit(args.output, args.csv, manifest)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return evidence

if __name__ == "__main__":
    main()
