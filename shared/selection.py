"""Configuration of the federated feature selection (Stage 4).

One federated run collects per-site statistics for the four dimensions once;
several *strategies* are then evaluated on those same statistics at the
coordinator, so comparing aggregation methods or weights costs no extra
communication.

No threshold, weight or size has a default: each is an experiment parameter
that has to be justified empirically.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Upper bound on the candidate pool of the redundancy round (message size:
#: the scatter matrix has p(p+1)/2 entries).
MAX_CANDIDATES = 200

_NAME = r"^[a-z0-9_]{1,48}$"
ReliabilityColumn = Literal["min_icc", "min_ci_low"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReliabilityCriterion(_Model):
    aggregation: Literal["minimum", "weighted_mean"]
    """How site values are combined: the worst site, or a mean weighted by the
    number of subjects in each site's reliability study."""
    min_value: float = Field(ge=-1, le=1)


class RelevanceCriterion(_Model):
    aggregation: Literal["stratified_score", "stouffer", "mean_rank"]
    """``stratified_score``: site-stratified Cox score test (sum U / sqrt(sum I)).
    ``stouffer``: per-site z combined with weights sqrt(events).
    ``mean_rank``: mean over sites of the per-site rank of |z| (rank-based,
    insensitive to the scale of each site's statistic)."""


class AgreementCriterion(_Model):
    method: Literal["i_squared", "sign_consistency"]
    """``i_squared``: agreement = 1 - I^2 of the one-step per-site estimates.
    ``sign_consistency``: fraction of sites whose effect sign matches the pooled sign."""
    min_value: float = Field(ge=0, le=1)


class SequentialRanking(_Model):
    """Filters on reliability/agreement, then ranks by relevance only."""

    kind: Literal["sequential"]


class DimensionWeights(_Model):
    reliability: float = Field(ge=0)
    relevance: float = Field(ge=0)
    agreement: float = Field(ge=0)

    @model_validator(mode="after")
    def _positive(self) -> DimensionWeights:
        if self.reliability + self.relevance + self.agreement <= 0:
            raise ValueError("at least one weight must be positive")
        return self


class WeightedRanking(_Model):
    """Ranks by a weighted mean of per-dimension percentile ranks.

    Provided to *compare* weightings; no weighting is preferred by the code.
    """

    kind: Literal["weighted"]
    weights: DimensionWeights


Ranking = Annotated[SequentialRanking | WeightedRanking, Field(discriminator="kind")]


class SelectionStrategy(_Model):
    name: str = Field(pattern=_NAME)
    reliability: ReliabilityCriterion | None
    """Reliability filter (null disables it explicitly)."""
    relevance: RelevanceCriterion
    agreement: AgreementCriterion | None
    """Between-site agreement filter (null disables it explicitly)."""
    ranking: Ranking
    max_abs_correlation: float | None = Field(gt=0, le=1)
    """Redundancy pruning threshold (null disables it explicitly)."""
    n_features: int = Field(ge=1)


class RedundancyRound(_Model):
    correlation: Literal["pearson", "spearman"]
    """``spearman`` ranks each feature within each site before pooling."""
    max_candidates: int = Field(ge=2, le=MAX_CANDIDATES)
    """Per strategy: how many top-ranked features enter the candidate pool."""


class FederatedSelectionConfig(_Model):
    reliability_column: ReliabilityColumn | None
    """Which local reliability summary sites report; null skips the round."""
    redundancy: RedundancyRound | None
    """Null skips the redundancy round."""
    strategies: tuple[SelectionStrategy, ...] = Field(min_length=1)
    leave_one_site_out: bool
    """Re-run every strategy without each site to measure selection stability."""

    @model_validator(mode="after")
    def _consistency(self) -> FederatedSelectionConfig:
        names = [s.name for s in self.strategies]
        if len(set(names)) != len(names):
            raise ValueError("strategy names must be unique")
        for strategy in self.strategies:
            uses_reliability = strategy.reliability is not None or (
                isinstance(strategy.ranking, WeightedRanking)
                and strategy.ranking.weights.reliability > 0
            )
            if uses_reliability and self.reliability_column is None:
                raise ValueError(
                    f"strategy {strategy.name!r} uses reliability but reliability_column is null"
                )
            if (
                isinstance(strategy.ranking, WeightedRanking)
                and strategy.ranking.weights.reliability > 0
                and strategy.reliability is None
            ):
                raise ValueError(
                    f"strategy {strategy.name!r}: weighting reliability needs a reliability "
                    "criterion to choose the aggregation (min_value -1 means no filtering)"
                )
            if (
                isinstance(strategy.ranking, WeightedRanking)
                and strategy.ranking.weights.agreement > 0
                and strategy.agreement is None
            ):
                raise ValueError(
                    f"strategy {strategy.name!r}: weighting agreement needs an agreement "
                    "criterion to choose the method (min_value 0 means no filtering)"
                )
            if strategy.max_abs_correlation is not None and self.redundancy is None:
                raise ValueError(
                    f"strategy {strategy.name!r} prunes redundancy but the redundancy round "
                    "is disabled"
                )
        if self.redundancy is not None:
            if len(self.strategies) * self.redundancy.max_candidates > MAX_CANDIDATES:
                raise ValueError(
                    f"strategies x max_candidates must not exceed {MAX_CANDIDATES} "
                    "(size of the covariance message)"
                )
            for strategy in self.strategies:
                if strategy.n_features > self.redundancy.max_candidates:
                    raise ValueError(f"strategy {strategy.name!r}: n_features > max_candidates")
        return self
