"""Released (global) prognostic models.

A released model contains only global quantities: coefficients, the
standardisation used, the pooled baseline and training diagnostics. It is what
the backend stores and what a site needs to compute predictions locally.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FitReport(_Model):
    converged: bool
    failure_reason: str | None = None
    iterations: int
    """Accepted Newton iterates."""
    evaluations: int
    """Log-likelihood evaluations (= communication rounds when federated)."""
    penalized_loglik_history: tuple[float, ...]
    n_clients: int
    n_samples: int
    n_events: int


class CoxModel(_Model):
    kind: Literal["cox"] = "cox"
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    """Log-hazard ratio per standard deviation of each feature."""
    standard_errors: tuple[float, ...]
    horizons_months: tuple[float, ...]
    baseline_cumulative_hazard: tuple[float | None, ...]
    """Breslow H0 at each horizon for standardised features (None = beyond follow-up)."""
    baseline: Literal["local", "pooled_sites"]
    """``pooled_sites``: average of the site-specific baselines (see docs)."""
    fit: FitReport


class HorizonFit(_Model):
    horizon_months: float
    intercept: float
    """Common intercept, or the size-weighted average of the site intercepts."""
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    """For the coefficients only."""
    fit: FitReport


class HorizonLogisticModel(_Model):
    kind: Literal["horizon_logistic"] = "horizon_logistic"
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    censoring: Literal["ipcw", "complete_case"]
    intercept: Literal["common", "site_specific"]
    horizons: tuple[HorizonFit, ...]


ReleasedModel = Annotated[CoxModel | HorizonLogisticModel, Field(discriminator="kind")]


class TrainedModel(_Model):
    """Outcome of training one configured model."""

    name: str
    status: Literal["trained", "failed"]
    failure_reason: str | None = None
    features_source: str
    model: ReleasedModel | None = None
