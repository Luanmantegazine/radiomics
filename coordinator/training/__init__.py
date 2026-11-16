"""Federated model training (Stage 5). Interface only in Stage 1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from shared.protocol import ClientUpdate


class TrainingStrategy(Protocol):
    """Model-specific aggregation of local training results.

    Survival models (e.g. Cox partial likelihood, whose risk sets span sites)
    generally need dedicated protocols rather than parameter averaging.
    """

    name: str

    def initial_state(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def aggregate(
        self, state: dict[str, Any], updates: Sequence[ClientUpdate]
    ) -> dict[str, Any]: ...

    def converged(self, previous: dict[str, Any], current: dict[str, Any]) -> bool: ...
