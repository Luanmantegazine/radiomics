"""Censoring conventions shared by models and metrics.

For a horizon tau, a participant is a **converter** (event at ``T <= tau``), a
**known non-converter** (``T > tau``) or of **unknown status** (censored at or
before ``tau``) - never a non-converter. Inverse-probability-of-censoring
weights use the Kaplan-Meier estimate G of the censoring distribution, with
conversions tied with censorings taken to happen first.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Censoring = Literal["ipcw", "complete_case"]


def censoring_survival(time: np.ndarray, event: np.ndarray):
    """Kaplan-Meier estimate of P(C > t), as a step function.

    Returns ``(times, values)``: G is ``values[k]`` on ``[times[k], times[k+1])``
    and 1 before ``times[0]``. Conversions tied with censorings are taken to
    happen first, so they are no longer at risk of censoring at that time.
    """
    time, censored = np.asarray(time, dtype=float), ~np.asarray(event, dtype=bool)
    times = np.unique(time[censored])
    values = np.empty(len(times))
    survival = 1.0
    for k, t in enumerate(times):
        at_risk = (time > t).sum() + ((time == t) & censored).sum()
        survival *= 1.0 - ((time == t) & censored).sum() / at_risk
        values[k] = survival
    return times, values


def _evaluate(times: np.ndarray, values: np.ndarray, t: np.ndarray, left: bool) -> np.ndarray:
    """G(t) (``left=False``) or G(t-) (``left=True``)."""
    index = np.searchsorted(times, t, side="left" if left else "right") - 1
    return np.where(index >= 0, values[np.clip(index, 0, None)], 1.0)


def horizon_labels(
    time: np.ndarray, event: np.ndarray, horizon: float, censoring: Censoring
) -> tuple[np.ndarray, np.ndarray]:
    """Labels and weights for one horizon (weight 0 = unknown status)."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    converter = event & (time <= horizon)
    known_negative = time > horizon
    y = converter.astype(float)
    if censoring == "complete_case":
        return y, (converter | known_negative).astype(float)
    times, values = censoring_survival(time, event)
    weights = np.zeros(len(time))
    g_event = _evaluate(times, values, time[converter], left=True)
    g_horizon = float(_evaluate(times, values, np.array([horizon]), left=False)[0])
    weights[converter] = 1.0 / g_event
    if g_horizon > 0:
        weights[known_negative] = 1.0 / g_horizon
    return y, weights
