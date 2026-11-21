"""Global selection strategies evaluated on pooled federated statistics.

Pipeline per strategy:

    exclusions -> reliability filter -> agreement filter -> ranking
               -> redundancy pruning (greedy, on the candidate pool) -> top k

Hypotheses and limitations
--------------------------
* Hard thresholds make the selection sensitive to small differences around
  them; thresholds and weights are experiment parameters, not defaults.
* ``weighted`` ranking combines percentile ranks, so a weight expresses the
  relative importance of *orderings*, not of effect sizes. Percentiles are
  computed among the features that passed the filters.
* Greedy pruning keeps the best-ranked member of each correlated group; the
  result depends on small ranking differences among correlated features.
* Redundancy is only known inside the candidate pool. When redundancy is
  pruned, features outside the pool are never selected (fewer than
  ``n_features`` may be returned).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from coordinator.selection import statistics as st
from shared.selection import SelectionStrategy, WeightedRanking


@dataclass(frozen=True, slots=True)
class PooledStatistics:
    """Everything a strategy needs, computed from one set of sites."""

    feature_names: tuple[str, ...]
    reliability: dict[str, np.ndarray] | None  # aggregation -> (p,)
    relevance: dict[str, np.ndarray]  # aggregation -> score (p,), larger = better
    relevance_z: dict[str, np.ndarray]  # aggregation -> signed z (p,) or NaN
    agreement: dict[str, np.ndarray]  # method -> (p,)
    estimable: np.ndarray  # (p,) bool
    correlation: np.ndarray | None = None  # (q, q) over the candidate pool
    pool: tuple[int, ...] | None = None  # feature indices of the candidate pool


def pooled_statistics(
    feature_names: tuple[str, ...],
    reliability_sites: Sequence[st.SiteReliability] | None,
    relevance_sites: Sequence[st.SiteRelevance],
) -> PooledStatistics:
    estimable = st.estimable(relevance_sites)
    stratified = st.stratified_z(relevance_sites)
    stouffer = st.stouffer_z(relevance_sites)
    return PooledStatistics(
        feature_names=feature_names,
        reliability=None
        if reliability_sites is None
        else {
            "minimum": st.reliability_minimum(reliability_sites),
            "weighted_mean": st.reliability_weighted_mean(reliability_sites),
        },
        relevance={
            "stratified_score": np.abs(stratified),
            "stouffer": np.abs(stouffer),
            "mean_rank": st.mean_rank_score(relevance_sites, estimable),
        },
        relevance_z={
            "stratified_score": stratified,
            "stouffer": stouffer,
            "mean_rank": np.full(len(feature_names), np.nan),
        },
        agreement={
            "i_squared": 1.0 - st.i_squared(relevance_sites),
            "sign_consistency": st.sign_consistency(relevance_sites),
        },
        estimable=estimable,
    )


@dataclass(slots=True)
class Ranked:
    """Outcome of the filter + ranking steps (before redundancy pruning)."""

    order: list[int]
    ranking_score: np.ndarray
    excluded: dict[str, int] = field(default_factory=dict)


def rank(strategy: SelectionStrategy, stats: PooledStatistics) -> Ranked:
    p = len(stats.feature_names)
    keep = stats.estimable.copy()
    excluded: dict[str, int] = {}

    def drop(mask: np.ndarray, reason: str) -> None:
        newly = keep & mask
        if newly.any():
            excluded[reason] = excluded.get(reason, 0) + int(newly.sum())
        keep[newly] = False

    excluded_not_estimable = int((~stats.estimable).sum())
    if excluded_not_estimable:
        excluded["not_estimable_at_all_sites"] = excluded_not_estimable
    reliability = None
    if strategy.reliability is not None:
        reliability = stats.reliability[strategy.reliability.aggregation]
        drop(~np.isfinite(reliability), "reliability_unknown")
        drop(np.nan_to_num(reliability, nan=-np.inf) < strategy.reliability.min_value,
             "below_reliability_threshold")
    agreement = None
    if strategy.agreement is not None:
        agreement = stats.agreement[strategy.agreement.method]
        drop(~np.isfinite(agreement), "agreement_undefined")
        drop(np.nan_to_num(agreement, nan=-np.inf) < strategy.agreement.min_value,
             "below_agreement_threshold")
    relevance = stats.relevance[strategy.relevance.aggregation]
    drop(~np.isfinite(relevance), "relevance_undefined")

    score = np.full(p, np.nan)
    eligible = np.flatnonzero(keep)
    if isinstance(strategy.ranking, WeightedRanking) and eligible.size:
        weights = strategy.ranking.weights
        parts = [(weights.relevance, relevance)]
        if weights.reliability > 0:
            parts.append((weights.reliability, reliability))
        if weights.agreement > 0:
            parts.append((weights.agreement, agreement))
        defined = np.all([np.isfinite(v[eligible]) for _, v in parts], axis=0)
        if not defined.all():
            drop(np.isin(np.arange(p), eligible[~defined]), "weighted_dimension_undefined")
            eligible = np.flatnonzero(keep)
        total = sum(w for w, _ in parts)
        score[eligible] = sum(
            w * st.percentile_rank(v[eligible]) for w, v in parts if w > 0
        ) / total
    else:
        score[eligible] = relevance[eligible]
    order = list(eligible[np.argsort(-score[eligible], kind="stable")])
    return Ranked(order=order, ranking_score=score, excluded=excluded)


def prune(
    strategy: SelectionStrategy, ranked: Ranked, stats: PooledStatistics
) -> list[int]:
    if strategy.max_abs_correlation is None:
        return ranked.order[: strategy.n_features]
    if stats.correlation is None or stats.pool is None:
        raise ValueError("redundancy pruning needs the pooled correlation of the candidate pool")
    position = {feature: i for i, feature in enumerate(stats.pool)}
    chosen: list[int] = []
    for feature in ranked.order:
        if feature not in position:
            continue
        if all(
            stats.correlation[position[feature], position[other]] < strategy.max_abs_correlation
            for other in chosen
        ):
            chosen.append(feature)
            if len(chosen) == strategy.n_features:
                break
    return chosen


def select(strategy: SelectionStrategy, stats: PooledStatistics) -> tuple[list[int], Ranked]:
    ranked = rank(strategy, stats)
    return prune(strategy, ranked, stats), ranked
