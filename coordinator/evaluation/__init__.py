"""Distributed evaluation (Stage 6). Interface only in Stage 1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from shared.protocol import ClientUpdate


class EvaluationStrategy(Protocol):
    """Combines per-site performance summaries into cohort-level comparisons."""

    name: str

    def combine(self, updates: Sequence[ClientUpdate], params: dict[str, Any]) -> Any: ...
