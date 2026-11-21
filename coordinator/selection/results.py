"""Released result of a federated selection run.

Contains global (pooled) statistics of the selected features and selection
sets only. Per-site statistics and leave-one-site-out *scores* are never part
of it: comparing a leave-one-out aggregate with the full one would reveal the
left-out site's statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SelectedFeature(_Model):
    rank: int = Field(ge=1)
    feature: str
    ranking_score: float
    reliability: float | None
    relevance_z: float | None
    """Signed pooled statistic of the strategy's relevance aggregation (None for
    ``mean_rank``, which has no z)."""
    relevance_p: float | None
    agreement: float | None


class LeaveOneSiteOut(_Model):
    status: str
    """``computed`` or the reason it was skipped."""
    selections: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Left-out client id -> selected features (no scores)."""
    nogueira_stability: float | None = None
    mean_jaccard_to_full: float | None = None


class StrategyResult(_Model):
    name: str
    completed: bool
    failure_reason: str | None = None
    selected: tuple[SelectedFeature, ...] = ()
    n_eligible: int = 0
    excluded_counts: dict[str, int] = Field(default_factory=dict)
    leave_one_site_out: LeaveOneSiteOut | None = None


class FederatedSelectionResult(_Model):
    n_clients: dict[str, int]
    """Round task -> number of sites whose statistics were used."""
    n_features_considered: int
    candidate_pool_size: int | None
    strategies: tuple[StrategyResult, ...]
    jaccard_between_strategies: dict[str, float]
