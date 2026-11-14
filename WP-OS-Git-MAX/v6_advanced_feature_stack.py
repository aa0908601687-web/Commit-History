#!/usr/bin/env python3
"""Optional V6 feature-engineering E2E: tsfresh + Featuretools -> exact selector.

Only the latest 10 completed draws are used to construct current candidate
features. This is a research feature pipeline, not a winning-probability model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _minmax(values):
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def build_features(rows):
    """Return engineered candidate features and transparent extraction metadata."""
    import pandas as pd
    import featuretools as ft
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute

    if len(rows) < 10:
        raise ValueError("advanced feature stack requires at least 10 completed draws")
    history = rows[-10:]
    records = []
    series = []
    for number in range(1, 39):
        binary = [int(number in draw[1]) for draw in history]
        recent5 = binary[-5:]
        weighted = sum((i + 1) * value for i, value in enumerate(binary)) / 55.0
        gap = next((age for age, value in enumerate(reversed(binary), 1) if value), 11)
        records.append({
            "number": number,
            "freq_5": float(sum(recent5)),
            "freq_10": float(sum(binary)),
            "weighted_freq_10": float(weighted),
            "gap_10": float(min(gap, 10)),
        })
        for time_index, value in enumerate(binary):
            series.append({"id": number, "time": time_index, "value": value})

    ts_input = pd.DataFrame(series)
    ts_features = extract_features(
        ts_input,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=0,
    )
    impute(ts_features)
    ts_features = ts_features.reset_index().rename(columns={"id": "number"})
    base = pd.DataFrame(records)
    combined = base.merge(ts_features, on="number", how="inner", validate="one_to_one")

    entity_set = ft.EntitySet(id="v6_candidate_features")
    entity_set = entity_set.add_dataframe(
        dataframe_name="numbers",
        dataframe=combined,
        index="number",
        make_index=False,
    )
    feature_matrix, feature_defs = ft.dfs(
        entityset=entity_set,
        target_dataframe_name="numbers",
        trans_primitives=["add_numeric", "subtract_numeric", "multiply_numeric"],
        max_depth=1,
        verbose=False,
    )
    numeric = feature_matrix.select_dtypes(include=["number"]).copy()
    numeric = numeric.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    if numeric.shape[0] != 38 or numeric.shape[1] < 5:
        raise RuntimeError("feature engineering produced an incomplete candidate matrix")

    normalized_columns = []
    for column in numeric.columns:
        values = [float(x) for x in numeric[column].tolist()]
        normalized_columns.append(_minmax(values))
    scores = {
        number: float(sum(column[index] for column in normalized_columns) / len(normalized_columns))
        for index, number in enumerate(range(1, 39))
    }
    if set(scores) != set(range(1, 39)) or not all(math.isfinite(v) for v in scores.values()):
        raise RuntimeError("advanced feature stack returned invalid candidate scores")
    metadata = {
        "enabled": True,
        "executed": True,
        "engine": "tsfresh+Featuretools",
        "feature_window": [5, 10],
        "current_feature_history_rows": 10,
        "candidate_count": 38,
        "tsfresh_calculators": "MinimalFCParameters",
        "tsfresh_feature_count": int(ts_features.shape[1] - 1),
        "featuretools_feature_count": int(len(feature_defs)),
        "generated_numeric_feature_count": int(numeric.shape[1]),
        "feature_names": [str(x) for x in numeric.columns],
        "leakage_safe_by_construction": True,
        "predictive_advantage_verified": False,
    }
    return scores, metadata


def run(csv_path):
    from v6_exact_e2e import validate, complete_result

    rows = validate(csv_path)
    scores, metadata = build_features(rows)
    result = complete_result(
        csv_path,
        rows,
        scores,
        {"enabled": True, "weight": 1.0, **metadata},
    )
    result["model"] = "tsfresh + Featuretools feature generation -> exact 8-choose-6 search; research only"
    result["advanced_feature_stack"] = metadata
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.csv)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
