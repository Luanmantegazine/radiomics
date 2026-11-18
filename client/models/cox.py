"""Cox proportional hazards: partial likelihood, Breslow baseline, local model.

Hypotheses and limitations
--------------------------
* Proportional hazards and log-linear effects of the (standardised) features.
* Ties are handled with Breslow's approximation.
* Non-informative right censoring.
* Strata (here: sites) have their own baseline hazard and share coefficients.
* The baseline cumulative hazard is not extrapolated: at a horizon after the
  last follow-up time it is undefined (None), never the last value carried on.
* Ridge penalty on standardised coefficients; no variable selection here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.models import CoxModel, FitReport
from shared.newton import NewtonOptimizer, NewtonSettings


def _validate(x: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
    if x.ndim != 2 or len(time) != len(x) or len(event) != len(x):
        raise ValueError("x must be (n, p) with one time and event per row")
    if not (np.isfinite(x).all() and np.isfinite(time).all() and (time > 0).all()):
        raise ValueError("features and times must be finite, times positive")


def partial_likelihood(
    x: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray,
    strata: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Breslow log partial likelihood, gradient and observed information.

    With ``strata``, risk sets are restricted to each stratum and the three
    quantities are summed over strata.
    """
    x, time = np.asarray(x, dtype=float), np.asarray(time, dtype=float)
    event, beta = np.asarray(event, dtype=bool), np.asarray(beta, dtype=float)
    _validate(x, time, event)
    if strata is not None:
        total = (0.0, np.zeros(len(beta)), np.zeros((len(beta), len(beta))))
        for stratum in np.unique(strata):
            rows = strata == stratum
            part = partial_likelihood(x[rows], time[rows], event[rows], beta)
            total = (total[0] + part[0], total[1] + part[1], total[2] + part[2])
        return total
    order = np.argsort(time, kind="stable")
    x, time, event = x[order], time[order], event[order]
    eta = x @ beta
    shift = eta.max() if len(eta) else 0.0
    w = np.exp(eta - shift)
    # Reverse cumulative sums = sums over risk sets {j : time_j >= t}.
    s0 = np.cumsum(w[::-1])[::-1]
    s1 = np.cumsum((w[:, None] * x)[::-1], axis=0)[::-1]
    s2 = np.cumsum((w[:, None, None] * x[:, :, None] * x[:, None, :])[::-1], axis=0)[::-1]
    loglik, gradient = 0.0, np.zeros(x.shape[1])
    information = np.zeros((x.shape[1], x.shape[1]))
    for t in np.unique(time[event]):
        start = int(np.searchsorted(time, t, side="left"))
        failing = event & (time == t)
        d = int(failing.sum())
        mean = s1[start] / s0[start]
        loglik += float(eta[failing].sum() - d * (np.log(s0[start]) + shift))
        gradient += x[failing].sum(axis=0) - d * mean
        information += d * (s2[start] / s0[start] - np.outer(mean, mean))
    return loglik, gradient, information


def breslow_cumulative_hazard(
    x: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray,
    horizons: tuple[float, ...],
) -> tuple[float | None, ...]:
    """Breslow H0 at each horizon; None when nobody is at risk at the horizon."""
    x, time = np.asarray(x, dtype=float), np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.exp(x @ np.asarray(beta, dtype=float))
    event_times = np.unique(time[event])
    increments = np.array(
        [((time == t) & event).sum() / risk[time >= t].sum() for t in event_times]
    )
    out: list[float | None] = []
    for horizon in horizons:
        if not (time >= horizon).any():
            out.append(None)
        else:
            out.append(float(increments[event_times <= horizon].sum()))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Standardization:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardization:
        scale = x.std(axis=0, ddof=1)
        if (scale <= 0).any() or not np.isfinite(scale).all():
            raise ValueError("a feature has no variation; remove it before fitting")
        return cls(x.mean(axis=0), scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.center) / self.scale


def fit_cox(
    feature_names: tuple[str, ...],
    x: np.ndarray, time: np.ndarray, event: np.ndarray,
    settings: NewtonSettings, horizons: tuple[float, ...],
    *, min_events: int,
) -> CoxModel:
    """Fit a (ridge-penalised) Cox model on one site's data."""
    n_events = int(np.sum(event))
    if n_events < min_events:
        raise ValueError(f"{n_events} events observed, at least {min_events} required")
    standardization = Standardization.fit(np.asarray(x, dtype=float))
    z = standardization.apply(x)
    optimizer = NewtonOptimizer(z.shape[1], settings)
    while not optimizer.finished:
        optimizer.observe(*partial_likelihood(z, time, event, optimizer.proposal))
    if optimizer.best is None:
        raise ValueError(f"Cox fit failed: {optimizer.failed}")
    return CoxModel(
        feature_names=feature_names,
        center=tuple(standardization.center.tolist()),
        scale=tuple(standardization.scale.tolist()),
        coefficients=tuple(optimizer.best.tolist()),
        standard_errors=tuple(optimizer.standard_errors().tolist()),
        horizons_months=horizons,
        baseline_cumulative_hazard=breslow_cumulative_hazard(
            z, time, event, optimizer.best, horizons
        ),
        baseline="local",
        fit=FitReport(
            converged=optimizer.converged,
            failure_reason=optimizer.failed,
            iterations=len(optimizer.history),
            evaluations=optimizer.evaluations,
            penalized_loglik_history=tuple(optimizer.history),
            n_clients=1,
            n_samples=len(time),
            n_events=n_events,
        ),
    )
