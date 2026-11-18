"""Configuration of harmonization and evaluation (Stage 6). No defaults."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HarmonizationConfig(_Model):
    method: Literal["none", "combat"]
    """``none`` only assesses site effects; ``combat`` also removes them."""
    covariates: tuple[str, ...]
    """Biological covariates whose effects ComBat must preserve (e.g. age).
    They are features of the agreed schema and are not themselves harmonized.
    The outcome is never allowed here."""
    empirical_bayes: bool
    eb_tolerance: float = Field(gt=0, lt=1)
    release_site_summaries: bool
    """Release per-site event rate and median follow-up (site-level values)."""


class EvaluationConfig(_Model):
    partition: Literal["test", "all"]
    """``test``: each site's held-out partition (needs ``data_split``)."""
    calibration_bins: int = Field(ge=2, le=100)
    """Equal-width bins of predicted risk on [0, 1]; small bins are merged."""
    decision_thresholds: tuple[float, ...]
    """Risk thresholds for net benefit; must be bin edges."""
    site_baseline_variant: bool
    """Also evaluate Cox models with each site's own baseline hazard and
    site-specific logistic models with each site's own intercept (estimated on
    the site's training partition)."""
    local_models: bool
    """Also evaluate, at each site, the same model trained on that site alone."""
    release_site_metrics: bool
    """Release per-site metrics (site-level performance) besides the pooled ones."""
    external_validation: Literal["none", "leave_one_site_out"]
    """Re-run selection and training without each site and evaluate on it."""

    @model_validator(mode="after")
    def _consistency(self) -> EvaluationConfig:
        edges = {round(k / self.calibration_bins, 12) for k in range(self.calibration_bins + 1)}
        if any(round(t, 12) not in edges or not 0 < t < 1 for t in self.decision_thresholds):
            raise ValueError("decision_thresholds must be interior calibration bin edges")
        if self.external_validation != "none" and not self.release_site_metrics:
            raise ValueError("external validation reports one site at a time: "
                             "it requires release_site_metrics")
        return self

    @property
    def grid(self) -> tuple[float, ...]:
        return tuple(round(k / self.calibration_bins, 12)
                     for k in range(self.calibration_bins + 1))
