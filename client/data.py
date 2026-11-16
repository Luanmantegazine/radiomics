"""Local, individual-level data. Objects defined here never leave the client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from shared.protocol import DataOrigin


@dataclass(frozen=True, slots=True)
class LocalDataset:
    """Per-participant observations held by one site.

    Each observation carries a follow-up time, an event indicator (right
    censoring when ``event`` is False) and a predictor vector. This type is
    intentionally *not* a Pydantic wire model and has no serialization method.
    """

    participant_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: np.ndarray  # shape (n_participants, n_features)
    time_months: np.ndarray  # shape (n_participants,)
    event: np.ndarray  # shape (n_participants,), bool; False = right-censored
    origin: DataOrigin

    def __post_init__(self) -> None:
        n = len(self.participant_ids)
        if self.features.shape != (n, len(self.feature_names)):
            raise ValueError("features must have shape (n_participants, n_features)")
        if self.time_months.shape != (n,) or self.event.shape != (n,):
            raise ValueError("time_months and event must have one entry per participant")

    @property
    def n_participants(self) -> int:
        return len(self.participant_ids)


class LocalDataSource(Protocol):
    """Loads the site's local dataset. Real sources arrive in Stage 2."""

    origin: DataOrigin

    def load(self) -> LocalDataset: ...
