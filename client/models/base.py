"""Common interface of local survival models (so algorithms can be swapped)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SurvivalModel(Protocol):
    name: str

    def fit(self, x: np.ndarray, time: np.ndarray, event: np.ndarray) -> SurvivalModel: ...

    def predict_conversion(self, x: np.ndarray) -> np.ndarray:
        """Probability of conversion by each configured horizon, shape (n, n_horizons)."""
        ...
