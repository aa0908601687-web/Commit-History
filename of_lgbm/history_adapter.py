"""Strict chronological Power Lottery CSV adapter for research only.

Expected columns: draw_id,date,n1,n2,n3,n4,n5,n6,second.
Dates must be ISO YYYY-MM-DD. Rows must be ascending by date/draw_id.
Never construct target-draw features from target-draw outcomes.
"""
from __future__ import annotations
import csv
from datetime import date
from pathlib import Path

FIELDS = ("draw_id", "date", "n1", "n2", "n3", "n4", "n5", "n6", "second")


def load_verified_format(path):
    """Validate a local CSV's *format*, not its official provenance."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("Missing required columns: " + ", ".join(FIELDS))
        rows = []
        seen = set()
        previous = None
        for row in reader:
            draw_id = row["draw_id"].strip()
            if not draw_id or draw_id in seen:
                raise ValueError("Duplicate or missing draw ID")
            seen.add(draw_id)
            try:
                draw_date = date.fromisoformat(row["date"].strip())
                first = tuple(int(row[f"n{i}"]) for i in range(1, 7))
                second = int(row["second"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid date or draw number") from exc
            if len(set(first)) != 6 or any(not 1 <= n <= 38 for n in first):
                raise ValueError("Invalid first-area numbers")
            if not 1 <= second <= 8:
                raise ValueError("Invalid second-area number")
            key = (draw_date, draw_id)
            if previous is not None and key <= previous:
                raise ValueError("Draws must be in strictly ascending order")
            previous = key
            rows.append({"draw_id": draw_id, "date": draw_date.isoformat(),
                         "first": tuple(sorted(first)), "second": second})
    if not rows:
        raise ValueError("Empty draw CSV")
    return rows


def historical_windows(rows, target_index):
    """Return only prior draws, with 10/20/all-history perspectives."""
    if not 0 < target_index <= len(rows):
        raise ValueError("Target index must have at least one prior draw")
    prior = rows[:target_index]
    return {"last_10": prior[-10:], "last_20": prior[-20:], "all_history": prior}
