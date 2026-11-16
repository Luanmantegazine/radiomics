"""Federated feature selection (Stage 4). Interface only in Stage 1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from shared.protocol import ClientUpdate


class FeatureSelectionStrategy(Protocol):
    """Combines per-site evidence into a global feature ranking/selection.

    Intended to weigh four configurable dimensions - measurement reliability,
    prognostic relevance, between-site agreement and redundancy - without
    hard-coding a formula or weights: those are experiment parameters to be
    justified empirically.
    """

    name: str

    def select(self, updates: Sequence[ClientUpdate], params: dict[str, Any]) -> Any: ...
