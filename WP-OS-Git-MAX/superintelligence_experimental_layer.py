"""Isolated compatibility layer for open-source AI experiments.

This module intentionally does not import or execute the upstream
``marciolscoutinho/superintelligence`` package.  The reviewed upstream commit
contains unresolved runtime references, so it is treated as a design reference
only.  Nothing in this module claims ASI capability or improved lottery odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


UPSTREAM_REPOSITORY = "https://github.com/marciolscoutinho/superintelligence"
UPSTREAM_REVIEWED_COMMIT = "da09a58"
LAYER_VERSION = "1.0.0-experimental"


class ExperimentalLayerDisabled(RuntimeError):
    """Raised when an experimental operation is requested without opt-in."""


@dataclass(frozen=True)
class AgentResult:
    name: str
    payload: Mapping[str, Any]


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def preprocess_draws(draws: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize completed Power Lottery draws.

    Required keys are ``draw_id``, ``first`` and ``second``.  First-zone values
    must contain six distinct integers in 1..38; second-zone values are 1..8.
    The function rejects duplicate draw IDs and preserves caller order.
    """

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in draws:
        draw_id = str(row["draw_id"])
        if draw_id in seen_ids:
            raise ValueError(f"duplicate draw_id: {draw_id}")
        seen_ids.add(draw_id)

        raw_first = row["first"]
        first = (
            [int(part) for part in raw_first.split()]
            if isinstance(raw_first, str)
            else [int(number) for number in raw_first]
        )
        second = int(row["second"])
        if len(first) != 6 or len(set(first)) != 6:
            raise ValueError("first zone must contain six distinct numbers")
        if any(number < 1 or number > 38 for number in first):
            raise ValueError("first-zone number outside 1..38")
        if second < 1 or second > 8:
            raise ValueError("second-zone number outside 1..8")
        normalized.append(
            {"draw_id": draw_id, "first": sorted(first), "second": second}
        )
    return normalized


class ExperimentalOrchestrator:
    """Small deterministic agent runner with explicit opt-in and audit output."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._agents: list[tuple[str, Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]]] = []

    def register(
        self,
        name: str,
        agent: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    ) -> None:
        if not name.strip():
            raise ValueError("agent name must not be empty")
        if any(existing == name for existing, _ in self._agents):
            raise ValueError(f"duplicate agent name: {name}")
        self._agents.append((name, agent))

    def run(self, draws: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        if not self.enabled:
            raise ExperimentalLayerDisabled(
                "experimental layer is disabled; pass enabled=True to opt in"
            )
        normalized = preprocess_draws(draws)
        results = [
            AgentResult(name=name, payload=dict(agent(normalized)))
            for name, agent in self._agents
        ]
        payload = {
            "layer_version": LAYER_VERSION,
            "status": "experimental",
            "asi_claim": False,
            "predictive_advantage_claim": False,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "reviewed_commit": UPSTREAM_REVIEWED_COMMIT,
                "code_imported": False,
            },
            "input": {
                "draw_count": len(normalized),
                "digest_sha256": _canonical_digest(normalized),
            },
            "agents": [
                {"name": result.name, "payload": result.payload}
                for result in results
            ],
        }
        payload["audit_sha256"] = _canonical_digest(payload)
        return payload

