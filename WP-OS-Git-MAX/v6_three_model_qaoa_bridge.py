"""Read-only three-model -> V6 exact-search comparison (not production inference)."""
import hashlib
import json
from pathlib import Path

from wp_os_git_max import load
from modular_selection_pipeline import run as run_three
from v6_ai_qaoa_e2e import run as run_exact


def run(csv_path):
    path = Path(csv_path)
    draws = load(str(path))
    trio = run_three(draws)
    exact = run_exact(str(path))
    cutoff = draws[-1]["draw_id"]
    if trio["through_draw"] != exact["last_draw"] or cutoff != exact["last_draw"]:
        raise ValueError("Exact-search and three-model cutoffs disagree")
    names = ("main_5_10", "control_10_20", "fusion_5_10_20")
    comparisons = {}
    efirst = set(exact["first_zone"])
    for name in names:
        prediction = trio["predictions"][name]
        first = prediction["first"]
        second = prediction["second"]
        if len(first) != 6 or len(set(first)) != 6 or not all(1 <= n <= 38 for n in first):
            raise ValueError("Invalid model first-zone output")
        if not isinstance(second, int) or not 1 <= second <= 8:
            raise ValueError("Invalid second-zone output")
        comparisons[name] = {
            "first_zone_overlap": sorted(efirst.intersection(first)),
            "first_zone_overlap_count": len(efirst.intersection(first)),
            "second_zone_equal": second == exact["second_zone"],
        }
    return {
        "cutoff": cutoff, "source_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "models": {name: {"first": trio["predictions"][name]["first"],
                           "second": trio["predictions"][name]["second"]} for name in names},
        "exact_search_research": {"first": exact["first_zone"], "second": exact["second_zone"]},
        "comparison": comparisons,
        "production_selection_unchanged": True,
        "live_official_api_verified": False,
        "trained_ml_verified": False,
        "predictive_advantage_verified": False,
        "full_e2e_verified": False,
        "note": "Offline comparison of existing archived data, not live API to trained AI deployment.",
    }


if __name__ == "__main__":
    archive = Path(__file__).with_name("historical_archive.csv")
    print(json.dumps(run(archive), sort_keys=True, ensure_ascii=False, indent=2))
