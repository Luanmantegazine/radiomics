"""Univariable prognostic relevance of features for time-to-event outcomes.

Both scorers use follow-up time and the event indicator, so right-censored
participants contribute exactly the information they carry (they are in the
risk set until their censoring time) and are never treated as non-converters.

``cox_score``
    Score (Rao) test of a univariable Cox model at beta = 0. Closed form, no
    iterative fitting, invariant to feature scale. Score = |z|.
``concordance``
    Harrell's C of the single feature. Score = 2 * |C - 0.5|, in [0, 1].

Hypotheses and limitations
--------------------------
* Univariable: no adjustment for age, sex, site or other features; a feature
  can rank high only because it proxies one of them.
* ``cox_score`` assumes proportional hazards and a log-linear effect; ties are
  handled with Breslow's approximation. ``concordance`` only assumes a
  monotone relation but Harrell's C is biased under heavy censoring.
* Non-informative censoring is assumed.
* p-values are per feature and unadjusted for multiplicity; with hundreds of
  features many small p-values are expected by chance.
* Relevance estimated on the same participants later used to evaluate a model
  leaks information: compute it inside the training split only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sksurv.metrics import concordance_index_censored


class RelevanceUnavailable(Exception):
    """Too little outcome information to score relevance."""


@dataclass(frozen=True, slots=True)
class RelevanceResult:
    method: str
    score: np.ndarray
    """Non-negative; larger = stronger univariable association. NaN = undefined."""
    statistic: np.ndarray
    """Signed statistic (z for ``cox_score``, C for ``concordance``)."""
    p_value: np.ndarray | None
    n_samples: int
    n_events: int


def _check(features: np.ndarray, time: np.ndarray, event: np.ndarray, min_events: int) -> None:
    if features.ndim != 2 or len(time) != len(features) or len(event) != len(features):
        raise ValueError("features must be (n, p) with one time and event per row")
    if not (np.isfinite(features).all() and np.isfinite(time).all() and (time > 0).all()):
        raise ValueError("features and times must be finite, times positive")
    n_events = int(np.sum(event))
    if n_events < min_events:
        raise RelevanceUnavailable(f"{n_events} events observed, at least {min_events} required")


def cox_score_components(
    features: np.ndarray, time: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Score U and information I of a univariable Cox model at beta = 0.

    Computed on the given risk sets only: summing (U, I) over sites gives the
    score test of a Cox model stratified by site. Breslow handling of ties.
    """
    x = np.asarray(features, dtype=float)
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    order = np.argsort(time, kind="stable")
    # U and I are invariant to shifting x; centring avoids cancellation in the
    # sum-of-squares formula for features with a large mean (e.g. volumes).
    x = x - x.mean(axis=0)
    x, time, event = x[order], time[order], event[order]
    n = len(time)
    # Suffix sums: totals over the risk set {j : time_j >= time_i}.
    risk_sum = np.cumsum(x[::-1], axis=0)[::-1]
    risk_sum_sq = np.cumsum((x**2)[::-1], axis=0)[::-1]
    score = np.zeros(x.shape[1])
    information = np.zeros(x.shape[1])
    for t in np.unique(time[event]):
        start = int(np.searchsorted(time, t, side="left"))
        at_risk = n - start
        failing = event & (time == t)
        d = int(failing.sum())
        mean = risk_sum[start] / at_risk
        score += x[failing].sum(axis=0) - d * mean
        information += d * np.clip(risk_sum_sq[start] / at_risk - mean**2, 0.0, None)
    return score, information


def cox_score_test(
    features: np.ndarray, time: np.ndarray, event: np.ndarray, *, min_events: int
) -> RelevanceResult:
    """Score test of H0: beta = 0 in a univariable Cox model, for every feature."""
    x = np.asarray(features, dtype=float)
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    _check(x, time, event, min_events)
    score, information = cox_score_components(x, time, event)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(information > 0, score / np.sqrt(information), np.nan)
    return RelevanceResult(
        method="cox_score",
        score=np.abs(z),
        statistic=z,
        p_value=2 * stats.norm.sf(np.abs(z)),
        n_samples=len(time),
        n_events=int(event.sum()),
    )


def concordance(
    features: np.ndarray, time: np.ndarray, event: np.ndarray, *, min_events: int
) -> RelevanceResult:
    """Harrell's concordance index of each feature taken alone."""
    x = np.asarray(features, dtype=float)
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    _check(x, time, event, min_events)
    c = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        if np.ptp(x[:, j]) > 0:
            c[j] = concordance_index_censored(event, time, x[:, j])[0]
    return RelevanceResult(
        method="concordance",
        score=2 * np.abs(c - 0.5),
        statistic=c,
        p_value=None,
        n_samples=len(time),
        n_events=int(event.sum()),
    )


RelevanceFunction = Callable[..., RelevanceResult]
RELEVANCE_METHODS: dict[str, RelevanceFunction] = {
    "cox_score": cox_score_test,
    "concordance": concordance,
}


def get_relevance(name: str) -> RelevanceFunction:
    try:
        return RELEVANCE_METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown relevance method {name!r}; available: {sorted(RELEVANCE_METHODS)}"
        ) from None
