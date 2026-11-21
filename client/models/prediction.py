"""Predicted probability of conversion by each horizon, from a released model.

Runs locally at a site (individual predictions never leave it).
"""

from __future__ import annotations

import numpy as np

from shared.models import CoxModel, HorizonLogisticModel, ReleasedModel


def _standardized(model, feature_names: tuple[str, ...], x: np.ndarray) -> np.ndarray:
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in model.feature_names if name not in index]
    if missing:
        raise ValueError(f"missing features: {missing}")
    columns = np.asarray(x, dtype=float)[:, [index[name] for name in model.feature_names]]
    return (columns - np.array(model.center)) / np.array(model.scale)


def linear_predictor(model: CoxModel, feature_names: tuple[str, ...], x: np.ndarray):
    return _standardized(model, feature_names, x) @ np.array(model.coefficients)


def predict_conversion(
    model: ReleasedModel, feature_names: tuple[str, ...], x: np.ndarray
) -> tuple[tuple[float, ...], np.ndarray]:
    """``(horizons, probabilities)`` with probabilities of shape (n, n_horizons).

    NaN where the model is undefined at a horizon (Cox baseline beyond follow-up).
    """
    z = _standardized(model, feature_names, x)
    if isinstance(model, CoxModel):
        risk = np.exp(z @ np.array(model.coefficients))
        h0 = np.array(
            [np.nan if v is None else v for v in model.baseline_cumulative_hazard], dtype=float
        )
        return model.horizons_months, 1.0 - np.exp(-np.outer(risk, h0))
    if isinstance(model, HorizonLogisticModel):
        columns = [
            1.0 / (1.0 + np.exp(-(fit.intercept + z @ np.array(fit.coefficients))))
            for fit in model.horizons
        ]
        return tuple(f.horizon_months for f in model.horizons), np.column_stack(columns)
    raise TypeError(f"unsupported model {type(model).__name__}")  # pragma: no cover
