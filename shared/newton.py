"""Penalised Newton-Raphson driven by externally evaluated log-likelihoods.

The optimiser never sees data: it receives the log-likelihood, gradient and
(observed) information evaluated at the point it proposed, wherever and however
they were computed. The same object therefore drives a local fit and a
federated fit in which every evaluation is a sum over sites.

Penalty: ``ridge / 2 * ||beta[penalized]||^2`` (an unpenalised intercept is
allowed). A step that lowers the penalised log-likelihood is halved, up to
``MAX_HALVINGS`` times, unless the decrease is within the convergence tolerance
(rounding at the optimum), which counts as convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_HALVINGS = 30


@dataclass(frozen=True, slots=True)
class NewtonSettings:
    ridge: float
    max_iterations: int
    tolerance: float
    """Convergence when the penalised log-likelihood changes by less than
    ``tolerance * (1 + |value|)`` between accepted iterates."""


@dataclass
class NewtonOptimizer:
    n_parameters: int
    settings: NewtonSettings
    penalized: np.ndarray | None = None
    proposal: np.ndarray = field(init=False)
    best: np.ndarray | None = field(init=False, default=None)
    best_value: float = field(init=False, default=-np.inf)
    best_information: np.ndarray | None = field(init=False, default=None)
    history: list[float] = field(init=False, default_factory=list)
    """Penalised log-likelihood of every accepted iterate."""
    evaluations: int = field(init=False, default=0)
    converged: bool = field(init=False, default=False)
    failed: str | None = field(init=False, default=None)
    _step: np.ndarray | None = field(init=False, default=None)
    _halvings: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.penalized is None:
            self.penalized = np.ones(self.n_parameters, dtype=bool)
        self.proposal = np.zeros(self.n_parameters)

    @property
    def finished(self) -> bool:
        return self.converged or self.failed is not None

    def observe(self, loglik: float, gradient: np.ndarray, information: np.ndarray) -> None:
        """Record the evaluation at ``self.proposal`` and choose the next proposal."""
        if self.finished:
            raise RuntimeError("optimiser already finished")
        self.evaluations += 1
        beta = self.proposal
        ridge = self.settings.ridge * self.penalized
        value = float(loglik - 0.5 * np.sum(ridge * beta**2))
        if not np.isfinite(value):
            value = -np.inf
        tolerance = self.settings.tolerance * (1 + abs(self.best_value))
        if self.best is not None and value < self.best_value:
            if self.best_value - value <= tolerance and self._halvings == 0:
                # A full Newton step that changes the objective by less than the
                # tolerance: we are at the optimum up to rounding.
                self.converged = True
                return
            self._halvings += 1
            if self._halvings > MAX_HALVINGS:
                self.failed = "step_halving_exhausted"
                return
            self.proposal = self.best + self._step * 0.5**self._halvings
            return
        previous = self.best_value
        gradient = np.asarray(gradient, dtype=float) - ridge * beta
        information = np.asarray(information, dtype=float) + np.diag(ridge)
        self.best, self.best_value, self.best_information = beta.copy(), value, information
        self.history.append(value)
        self._halvings = 0
        if np.isfinite(previous) and abs(value - previous) <= self.settings.tolerance * (
            1 + abs(value)
        ):
            self.converged = True
            return
        if len(self.history) >= self.settings.max_iterations:
            self.failed = "max_iterations"
            return
        try:
            self._step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            self.failed = "singular_information"
            return
        self.proposal = self.best + self._step

    def standard_errors(self) -> np.ndarray:
        """From the inverse penalised information at the best iterate."""
        if self.best_information is None:
            raise RuntimeError("no accepted iterate")
        return np.sqrt(np.clip(np.diag(np.linalg.inv(self.best_information)), 0.0, None))
