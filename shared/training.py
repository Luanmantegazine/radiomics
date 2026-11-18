"""Configuration of prognostic models (Stage 5).

The prediction horizons, the data split and every hyper-parameter are explicit
configuration; nothing has a default.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_NAME = r"^[a-z0-9_]{1,48}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSplitSpec(_Model):
    """Per-site split into a training and a held-out test partition.

    Feature selection and model fitting only ever see the training partition;
    the test partition is reserved for evaluation (Stage 6).
    """

    test_fraction: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)


class OptimizerSpec(_Model):
    ridge: float = Field(ge=0)
    """L2 penalty on standardised coefficients (intercepts are not penalised)."""
    max_iterations: int = Field(ge=1, le=100)
    tolerance: float = Field(gt=0, lt=1)


class CoxSpec(_Model):
    kind: Literal["cox"]
    name: str = Field(pattern=_NAME)
    optimizer: OptimizerSpec


class HorizonLogisticSpec(_Model):
    """Comparator: one logistic regression per horizon.

    ``ipcw`` weights participants by the inverse probability of remaining
    uncensored; ``complete_case`` drops those censored before the horizon
    (known to be biased, kept as a reference). Neither ever labels a
    participant censored at or before the horizon as a non-converter.
    """

    kind: Literal["horizon_logistic"]
    name: str = Field(pattern=_NAME)
    censoring: Literal["ipcw", "complete_case"]
    intercept: Literal["common", "site_specific"]
    """``site_specific`` gives each site its own intercept (the analogue of the
    site-stratified Cox model), so differences in feature levels between sites
    are not mistaken for effects; the coefficients are shared."""
    optimizer: OptimizerSpec


class RandomSurvivalForestSpec(_Model):
    """Example of another survival model behind the same interface (local only)."""

    kind: Literal["random_survival_forest"]
    name: str = Field(pattern=_NAME)
    n_estimators: int = Field(ge=10)
    min_samples_leaf: int = Field(ge=1)
    seed: int = Field(ge=0)


ModelSpec = Annotated[
    CoxSpec | HorizonLogisticSpec | RandomSurvivalForestSpec, Field(discriminator="kind")
]
FEDERATED_KINDS = {"cox", "horizon_logistic"}


class FromStrategy(_Model):
    from_strategy: str = Field(pattern=_NAME)


class ExplicitFeatures(_Model):
    names: tuple[str, ...] = Field(min_length=1)


class ModelEntry(_Model):
    model: ModelSpec
    features: FromStrategy | ExplicitFeatures


class TrainingConfig(_Model):
    horizons_months: tuple[float, ...] = Field(min_length=1)
    models: tuple[ModelEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistency(self) -> TrainingConfig:
        horizons = self.horizons_months
        if any(h <= 0 for h in horizons) or list(horizons) != sorted(set(horizons)):
            raise ValueError("horizons_months must be positive, unique and increasing")
        names = [entry.model.name for entry in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        return self
