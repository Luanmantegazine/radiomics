"""Versioned configuration of a local selection study.

Every threshold is a required field: there are no built-in defaults, because
none of these values is justified a priori. They are experiment parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RelevanceMethod = Literal["cox_score", "concordance"]
Correlation = Literal["spearman", "pearson"]
_NAME = r"^[a-z0-9_]{1,48}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReliabilityFilter(_Model):
    column: Literal["min_icc", "min_ci_low"]
    """Reliability summary column to threshold (point estimate or CI lower bound)."""
    min_value: float = Field(ge=-1, le=1)


class RedundancyFilter(_Model):
    correlation: Correlation
    max_abs_correlation: float = Field(gt=0, le=1)


class ComponentSelector(_Model):
    """Reliability filter -> relevance ranking -> redundancy pruning -> top k.

    The three components are applied as successive, separately configurable
    steps rather than blended into one weighted score, so no weights have to be
    invented. Either filter can be disabled by setting it to null.
    """

    method: Literal["component"]
    name: str = Field(pattern=_NAME)
    reliability: ReliabilityFilter | None
    relevance: RelevanceMethod
    redundancy: RedundancyFilter | None
    n_features: int = Field(ge=1)


class CorrelationFilterSelector(_Model):
    """Reference: univariable ranking with greedy correlation pruning."""

    method: Literal["correlation_filter"]
    name: str = Field(pattern=_NAME)
    relevance: RelevanceMethod
    correlation: Correlation
    max_abs_correlation: float = Field(gt=0, le=1)
    n_features: int = Field(ge=1)


class LassoCoxSelector(_Model):
    """Reference: L1-penalised Cox regression (regularisation path)."""

    method: Literal["lasso_cox"]
    name: str = Field(pattern=_NAME)
    n_features: int = Field(ge=1)
    n_alphas: int = Field(ge=10)
    alpha_min_ratio: float = Field(gt=0, lt=1)


class GraphSelector(_Model):
    """Reference: one representative per connected group of the correlation graph."""

    method: Literal["graph_representatives"]
    name: str = Field(pattern=_NAME)
    relevance: RelevanceMethod
    correlation: Correlation
    edge_threshold: float = Field(gt=0, le=1)
    n_features: int = Field(ge=1)


SelectorSpec = Annotated[
    ComponentSelector | CorrelationFilterSelector | LassoCoxSelector | GraphSelector,
    Field(discriminator="method"),
]


class BootstrapSpec(_Model):
    n_resamples: int = Field(ge=2)
    seed: int = Field(ge=0)


class SelectionStudyConfig(_Model):
    schema_version: Literal[1]
    study_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
    min_events: int = Field(ge=2)
    """Refuse outcome-based selection with fewer observed conversions."""
    selectors: tuple[SelectorSpec, ...] = Field(min_length=1)
    bootstrap: BootstrapSpec | None

    @model_validator(mode="after")
    def _unique(self) -> SelectionStudyConfig:
        names = [s.name for s in self.selectors]
        if len(set(names)) != len(names):
            raise ValueError("selector names must be unique")
        return self

    @property
    def needs_reliability(self) -> set[str]:
        return {
            s.reliability.column
            for s in self.selectors
            if isinstance(s, ComponentSelector) and s.reliability is not None
        }


def load_selection_config(path: str | Path) -> SelectionStudyConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return SelectionStudyConfig.model_validate(yaml.safe_load(handle))
