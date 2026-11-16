"""Stability of feature selection across repetitions (resamples, sites, rounds)."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np


def jaccard(a: Collection[str], b: Collection[str]) -> float:
    """Jaccard similarity of two feature sets (1.0 when both are empty)."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def selection_frequency(selections: np.ndarray) -> np.ndarray:
    """Fraction of repetitions in which each feature was selected."""
    return np.asarray(selections, dtype=bool).mean(axis=0)


def nogueira_stability(selections: np.ndarray) -> float:
    """Stability estimator of Nogueira, Sechidis & Brown (JMLR 2018).

    ``selections`` is a boolean matrix (repetitions x features). The value is
    at most 1 (identical selections) and has expectation 0 when features are
    selected at random; it is corrected for chance and valid when the number
    of selected features varies between repetitions.

    Limitation: it treats features as exchangeable labels, so swapping one
    feature for a highly correlated one counts as full disagreement.
    Returns NaN when undefined (fewer than two repetitions, or every repetition
    selecting none or all of the features).
    """
    z = np.asarray(selections, dtype=bool)
    if z.ndim != 2:
        raise ValueError("selections must be a (repetitions, features) matrix")
    m, p = z.shape
    if m < 2:
        return float("nan")
    frequency = z.mean(axis=0)
    k_bar = z.sum(axis=1).mean()
    denominator = (k_bar / p) * (1 - k_bar / p)
    if denominator <= 0:
        return float("nan")
    variance = m / (m - 1) * frequency * (1 - frequency)
    return float(1 - variance.mean() / denominator)
