"""Horizon-specific logistic regression with censoring handled explicitly.

For a horizon tau, a participant is

* a **converter** if the event was observed at ``T <= tau``;
* a **known non-converter** if followed up beyond ``tau`` (``T > tau``);
* of **unknown status** if censored at or before ``tau`` - never a non-converter.

``ipcw``: converters get weight 1/G(T-) and known non-converters 1/G(tau),
where G is the site's Kaplan-Meier estimate of the censoring distribution;
unknown-status participants get weight 0 but are represented through the
weights of the others. With these conventions the intercept-only IPCW estimate
equals the Kaplan-Meier cumulative incidence exactly (Satten & Datta 2001).
``complete_case``: unknown status is simply dropped
(weights 1); this is biased towards converters and kept only as a reference.

Hypotheses and limitations
--------------------------
* IPCW assumes censoring independent of conversion and of the features (G does
  not depend on covariates). Large weights (G close to 0) make it unstable.
* Each horizon is a separate model: predicted probabilities need not increase
  with the horizon.
* The intercept is shared by all sites (no site-specific baseline risk).
"""

from __future__ import annotations

import numpy as np

from client.models.cox import Standardization
from evaluation.censoring import (  # noqa: F401 - re-exported for callers
    Censoring,
    _evaluate,
    censoring_survival,
    horizon_labels,
)
from shared.models import FitReport, HorizonFit, HorizonLogisticModel
from shared.newton import NewtonOptimizer, NewtonSettings


def weighted_logistic(
    design: np.ndarray, y: np.ndarray, weights: np.ndarray, coefficients: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted Bernoulli log-likelihood, gradient and information.

    ``design`` includes the intercept column.
    """
    eta = design @ coefficients
    p = _sigmoid(eta)
    loglik = float(np.sum(weights * (y * eta - np.logaddexp(0.0, eta))))
    gradient = design.T @ (weights * (y - p))
    information = (design * (weights * p * (1 - p))[:, None]).T @ design
    return loglik, gradient, information


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -eta))


def profile_intercept(
    z: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: np.ndarray,
    tolerance: float = 1e-12, max_iterations: int = 200,
) -> float:
    """Maximum-likelihood intercept for fixed slopes.

    One-dimensional Newton with step halving on a concave objective; requires
    both outcome classes with positive weight (otherwise the maximum is at
    infinity).
    """
    used = weights > 0
    if not ((y[used] == 1).any() and (y[used] == 0).any()):
        raise ValueError("intercept not identifiable (one outcome class only)")
    offset = z @ beta

    def objective(alpha: float) -> float:
        eta = alpha + offset
        return float(np.sum(weights * (y * eta - np.logaddexp(0.0, eta))))

    alpha = 0.0
    value = objective(alpha)
    for _ in range(max_iterations):
        p = _sigmoid(alpha + offset)
        gradient = float(np.sum(weights * (y - p)))
        information = float(np.sum(weights * p * (1 - p)))
        step = gradient / information if information > 1e-300 else np.sign(gradient)
        for _ in range(60):
            candidate = objective(alpha + step)
            if candidate >= value:
                break
            step /= 2
        alpha, value = alpha + step, candidate
        if abs(step) < tolerance * (1 + abs(alpha)):
            return alpha
    raise ValueError("intercept did not converge")


def profile_logistic(
    z: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Profile log-likelihood of the slopes with a site-specific intercept.

    Returns ``(loglik, gradient, information, intercept)`` where gradient and
    information are those of the profile likelihood in ``beta`` (Schur
    complement of the intercept block). Summing them over sites maximises the
    joint likelihood with one intercept per site and shared slopes exactly.
    """
    alpha = profile_intercept(z, y, weights, beta)
    design = np.column_stack([np.ones(len(z)), z])
    loglik, gradient, information = weighted_logistic(design, y, weights, np.r_[alpha, beta])
    cross = information[1:, 0]
    profiled = information[1:, 1:] - np.outer(cross, cross) / information[0, 0]
    return loglik, gradient[1:], profiled, alpha


def counts(y: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    """(converters, known non-converters) with positive weight."""
    used = weights > 0
    return int((used & (y == 1)).sum()), int((used & (y == 0)).sum())


def fit_horizon_logistic(
    feature_names: tuple[str, ...],
    x: np.ndarray, time: np.ndarray, event: np.ndarray,
    horizons: tuple[float, ...], censoring: Censoring, settings: NewtonSettings,
    *, min_events: int,
) -> HorizonLogisticModel:
    standardization = Standardization.fit(np.asarray(x, dtype=float))
    design = np.column_stack([np.ones(len(time)), standardization.apply(x)])
    penalized = np.r_[False, np.ones(design.shape[1] - 1, dtype=bool)]
    fits = []
    for horizon in horizons:
        y, weights = horizon_labels(time, event, horizon, censoring)
        positives, negatives = counts(y, weights)
        if min(positives, negatives) < min_events:
            raise ValueError(f"horizon {horizon}: too few converters or known non-converters")
        optimizer = NewtonOptimizer(design.shape[1], settings, penalized)
        while not optimizer.finished:
            optimizer.observe(*weighted_logistic(design, y, weights, optimizer.proposal))
        if optimizer.best is None:
            raise ValueError(f"horizon {horizon}: fit failed ({optimizer.failed})")
        errors = optimizer.standard_errors()
        fits.append(
            HorizonFit(
                horizon_months=horizon,
                intercept=float(optimizer.best[0]),
                coefficients=tuple(optimizer.best[1:].tolist()),
                standard_errors=tuple(errors[1:].tolist()),
                fit=FitReport(
                    converged=optimizer.converged, failure_reason=optimizer.failed,
                    iterations=len(optimizer.history), evaluations=optimizer.evaluations,
                    penalized_loglik_history=tuple(optimizer.history), n_clients=1,
                    n_samples=positives + negatives, n_events=positives,
                ),
            )
        )
    return HorizonLogisticModel(
        feature_names=feature_names,
        center=tuple(standardization.center.tolist()),
        scale=tuple(standardization.scale.tolist()),
        censoring=censoring,
        intercept="common",  # one site: common and site-specific coincide
        horizons=tuple(fits),
    )
