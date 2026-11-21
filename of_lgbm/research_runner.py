"""Opt-in end-to-end research runner for normalized historical Power Lottery CSV.

No official-source authentication or claims of predictive advantage.
"""
from __future__ import annotations

from pathlib import Path

from feature_builder import build_number_features, build_supervised_history
from history_adapter import load_verified_format
from model_bridge import train_and_select
from pipeline import sha256_file


def run_research_csv(csv_path, *, min_history=20, seed=42, n_jobs=1,
                     trainer=None):
    """Train on historical target draws and select a hypothetical next draw.

    The model receives no unknown future outcome labels: validation labels
    are zero placeholders because the legacy trainer checks their lengths.
    """
    if not isinstance(min_history, int) or isinstance(min_history, bool) or min_history < 2:
        raise ValueError("min_history must be an integer >= 2")
    path = Path(csv_path)
    draws = load_verified_format(path)
    if len(draws) <= min_history:
        raise ValueError("Insufficient historical draws")
    import pandas as pd
    indices = range(1, len(draws))
    first_x, first_y = build_supervised_history(draws, indices, "first")
    second_x, second_y = build_supervised_history(draws, indices, "second")
    first_valid = build_number_features(draws, len(draws), "first")
    second_valid = build_number_features(draws, len(draws), "second")
    kwargs = {"first_numbers": list(range(1, 39)),
              "second_numbers": list(range(1, 9)), "seed": seed, "n_jobs": n_jobs}
    if trainer is not None:
        kwargs["trainer"] = trainer
    result = train_and_select(
        pd.DataFrame(first_x), first_y, pd.DataFrame(first_valid), [0] * 38,
        pd.DataFrame(second_x), second_y, pd.DataFrame(second_valid), [0] * 8,
        **kwargs,
    )
    return {
        "selection": result,
        "history_draws": len(draws),
        "last_draw_id": draws[-1]["draw_id"],
        "source_sha256": sha256_file(str(path)),
        "validation": "not a held-out or forward prediction evaluation",
    }
