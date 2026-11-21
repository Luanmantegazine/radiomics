"""Provisional evaluation contracts (reviewed when Stage 6 starts)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

#: Prediction horizons, in months, fixed by the study protocol.
HORIZONS_MONTHS: tuple[int, ...] = (12, 24, 36)


class SurvivalMetric(Protocol):
    """A censoring-aware performance metric.

    Implementations must treat participants without an observed conversion and
    with incomplete follow-up as *censored*, never as non-converters.
    """

    name: str

    def compute(
        self,
        time_months: np.ndarray,
        event: np.ndarray,
        predicted_risk: np.ndarray,
        horizon_months: int,
    ) -> float: ...
