"""Federated feature-selection protocol (coordinator side).

Communication rounds, each answered by every registered site:

1. ``feature_inventory``   - agree on the common feature schema (skipped when
                              an earlier round already agreed one).
2. ``feature_reliability`` - per-site reliability summary (skipped when no
                              strategy uses reliability). Sites without a
                              reliability study refuse; the others are used.
3. ``feature_relevance``   - per-site Cox score components (U, I).
4. ``feature_covariance``  - per-site within-site scatter of a candidate pool
                              (skipped when no strategy prunes redundancy).

Then every configured strategy is evaluated on the same statistics at the
coordinator (no extra communication), optionally with leave-one-site-out
re-evaluation to measure stability.

Per-site statistics are held in memory only for the duration of the run (the
strategy comparison and leave-one-site-out need them); they are not persisted
or released.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from coordinator.aggregation import AggregationError, SchemaIntersection
from coordinator.selection import statistics as st
from coordinator.selection.results import (
    FederatedSelectionResult,
    LeaveOneSiteOut,
    SelectedFeature,
    StrategyResult,
)
from coordinator.selection.strategies import (
    PooledStatistics,
    pooled_statistics,
    rank,
    select,
)
from evaluation.stability import jaccard, nogueira_stability
from shared.protocol import (
    ClientUpdate,
    FeatureCorrelationAggregate,
    FeatureCovariancePayload,
    FeatureRelevanceAggregate,
    FeatureRelevancePayload,
    FeatureReliabilityAggregate,
    FeatureReliabilityPayload,
    FederatedPhase,
    ModelState,
    PrivacyPolicy,
    TaskKind,
    WireModel,
)
from shared.selection import FederatedSelectionConfig


@dataclass(frozen=True, slots=True)
class RoundRequest:
    """One communication round requested by a protocol (selection or training)."""

    phase: FederatedPhase
    task: TaskKind
    feature_names: tuple[str, ...] | None
    params: dict
    aggregator: str
    model: ModelState | None = None


def _optional(values: np.ndarray) -> tuple[float | None, ...]:
    return tuple(float(v) if np.isfinite(v) else None for v in values)


class FederatedSelectionProtocol:
    """State machine driven by the coordinator engine."""

    def __init__(
        self,
        config: FederatedSelectionConfig,
        privacy: PrivacyPolicy,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self._config = config
        self._privacy = privacy
        self._features = feature_names
        self._reliability: list[st.SiteReliability] | None = None
        self._relevance: list[st.SiteRelevance] | None = None
        self._scatter: list[st.SiteScatter] | None = None
        self._pool: tuple[int, ...] | None = None
        self._done: set[TaskKind] = set()

    # -- round planning ------------------------------------------------------ #
    def next_request(self) -> RoundRequest | None:
        phase = FederatedPhase.FEATURE_SELECTION
        if self._features is None:
            return RoundRequest(phase, TaskKind.FEATURE_INVENTORY, None, {}, "schema_intersection")
        if (
            self._config.reliability_column is not None
            and TaskKind.FEATURE_RELIABILITY not in self._done
        ):
            return RoundRequest(
                phase, TaskKind.FEATURE_RELIABILITY, self._features,
                {"reliability_column": self._config.reliability_column}, "reliability",
            )
        if TaskKind.FEATURE_RELEVANCE not in self._done:
            return RoundRequest(
                phase, TaskKind.FEATURE_RELEVANCE, self._features, {}, "stratified_relevance"
            )
        if self._config.redundancy is not None and TaskKind.FEATURE_COVARIANCE not in self._done:
            self._pool = self._candidate_pool()
            if len(self._pool) < 2:
                self._done.add(TaskKind.FEATURE_COVARIANCE)  # nothing to correlate
                return None
            return RoundRequest(
                phase, TaskKind.FEATURE_COVARIANCE,
                tuple(self._features[i] for i in self._pool),
                {"rank_transform": self._config.redundancy.correlation == "spearman"},
                "pooled_within_site_correlation",
            )
        return None

    def on_round_failed(self, request: RoundRequest, reason: str) -> bool:
        """Every selection round is required: a failure stops the selection."""
        return False

    def aggregate(self, request: RoundRequest, updates: Sequence[ClientUpdate]) -> WireModel:
        """Combine the accepted updates of a round (engine already checked the count)."""
        self._done.add(request.task)
        if request.task is TaskKind.FEATURE_INVENTORY:
            aggregate = SchemaIntersection().aggregate(updates)
            if not aggregate.feature_names:
                raise AggregationError("the sites share no feature")
            self._features = aggregate.feature_names
            return aggregate
        if request.task is TaskKind.FEATURE_RELIABILITY:
            return self._aggregate_reliability(updates)
        if request.task is TaskKind.FEATURE_RELEVANCE:
            return self._aggregate_relevance(updates)
        if request.task is TaskKind.FEATURE_COVARIANCE:
            return self._aggregate_covariance(updates)
        raise AggregationError(f"unexpected task {request.task}")  # pragma: no cover

    def _payloads(self, updates, kind):
        payloads = [(u.client_id, u.payload) for u in updates if isinstance(u.payload, kind)]
        if len(payloads) != len(updates):
            raise AggregationError(f"expected {kind.__name__} payloads")
        return payloads

    def _aggregate_reliability(self, updates) -> FeatureReliabilityAggregate:
        payloads = self._payloads(updates, FeatureReliabilityPayload)
        self._reliability = [
            st.SiteReliability(
                client_id, p.n_subjects,
                np.array([np.nan if v is None else v for v in p.values], dtype=float),
            )
            for client_id, p in payloads
        ]
        return FeatureReliabilityAggregate(
            n_clients=len(payloads),
            column=self._config.reliability_column,
            feature_names=self._features,
            minimum=_optional(st.reliability_minimum(self._reliability)),
            weighted_mean=_optional(st.reliability_weighted_mean(self._reliability)),
        )

    def _aggregate_relevance(self, updates) -> FeatureRelevanceAggregate:
        payloads = self._payloads(updates, FeatureRelevancePayload)
        self._relevance = [
            st.SiteRelevance(
                client_id, p.n_samples, p.n_events,
                np.array(p.score, dtype=float), np.array(p.information, dtype=float),
            )
            for client_id, p in payloads
        ]
        return FeatureRelevanceAggregate(
            n_clients=len(payloads),
            n_samples=sum(p.n_samples for _, p in payloads),
            n_events=sum(p.n_events for _, p in payloads),
            feature_names=self._features,
            stratified_z=_optional(st.stratified_z(self._relevance)),
            i_squared=_optional(st.i_squared(self._relevance)),
            sign_consistency=_optional(st.sign_consistency(self._relevance)),
        )

    def _aggregate_covariance(self, updates) -> FeatureCorrelationAggregate:
        payloads = self._payloads(updates, FeatureCovariancePayload)
        q = len(self._pool)
        expected_rank = self._config.redundancy.correlation == "spearman"
        if any(p.rank_transformed != expected_rank for _, p in payloads):
            raise AggregationError("a site used a different correlation transform")
        self._scatter = [
            st.SiteScatter(client_id, p.n_samples, st.unpack_upper(p.scatter_upper, q))
            for client_id, p in payloads
        ]
        correlation = st.pooled_correlation(self._scatter)
        return FeatureCorrelationAggregate(
            n_clients=len(payloads),
            n_samples=sum(p.n_samples for _, p in payloads),
            method=self._config.redundancy.correlation,
            feature_names=tuple(self._features[i] for i in self._pool),
            correlation_upper=_optional(correlation[np.triu_indices(q)]),
        )

    # -- selection ----------------------------------------------------------- #
    def _with_correlation(self, stats: PooledStatistics, exclude: str | None) -> PooledStatistics:
        if self._scatter is None:
            return stats
        sites = [s for s in self._scatter if s.client_id != exclude]
        return replace(stats, correlation=st.pooled_correlation(sites), pool=self._pool)

    def _candidate_pool(self) -> tuple[int, ...]:
        """Union of each strategy's top ``max_candidates`` features (before pruning)."""
        stats = pooled_statistics(self._features, self._reliability, self._relevance)
        limit = self._config.redundancy.max_candidates
        pool: set[int] = set()
        for strategy in self._config.strategies:
            pool.update(rank(strategy, stats).order[:limit])
        return tuple(sorted(pool))

    def result(self) -> FederatedSelectionResult:
        if self._relevance is None:
            raise AggregationError("the relevance round did not complete")
        full = self._with_correlation(pooled_statistics(
            self._features, self._reliability, self._relevance), None)
        loso_sites = [s.client_id for s in self._relevance]
        strategies: list[StrategyResult] = []
        selections: dict[str, set[str]] = {}
        for strategy in self._config.strategies:
            try:
                chosen, ranked = select(strategy, full)
            except ValueError as exc:
                strategies.append(
                    StrategyResult(name=strategy.name, completed=False, failure_reason=str(exc))
                )
                continue
            names = [self._features[i] for i in chosen]
            selections[strategy.name] = set(names)
            strategies.append(
                StrategyResult(
                    name=strategy.name,
                    completed=True,
                    selected=tuple(
                        self._selected_feature(rank_, i, strategy, ranked, full)
                        for rank_, i in enumerate(chosen, start=1)
                    ),
                    n_eligible=len(ranked.order),
                    excluded_counts=dict(sorted(ranked.excluded.items())),
                    leave_one_site_out=self._loso(strategy, names, loso_sites),
                )
            )
        n_clients = {"feature_relevance": len(self._relevance)}
        if self._reliability is not None:
            n_clients["feature_reliability"] = len(self._reliability)
        if self._scatter is not None:
            n_clients["feature_covariance"] = len(self._scatter)
        return FederatedSelectionResult(
            n_clients=n_clients,
            n_features_considered=len(self._features),
            candidate_pool_size=None if self._pool is None else len(self._pool),
            strategies=tuple(strategies),
            jaccard_between_strategies={
                f"{a}|{b}": jaccard(selections[a], selections[b])
                for a, b in combinations(selections, 2)
            },
        )

    def _selected_feature(self, rank_, index, strategy, ranked, stats) -> SelectedFeature:
        def value(array):
            return None if array is None or not np.isfinite(array[index]) else float(array[index])

        reliability = (
            stats.reliability[strategy.reliability.aggregation]
            if strategy.reliability is not None and stats.reliability is not None
            else None
        )
        agreement = (
            stats.agreement[strategy.agreement.method] if strategy.agreement is not None else None
        )
        z = stats.relevance_z[strategy.relevance.aggregation]
        return SelectedFeature(
            rank=rank_,
            feature=self._features[index],
            ranking_score=float(ranked.ranking_score[index]),
            reliability=value(reliability),
            relevance_z=value(z),
            relevance_p=value(st.two_sided_p(z)),
            agreement=value(agreement),
        )

    def _loso(self, strategy, full_selection, sites) -> LeaveOneSiteOut | None:
        if not self._config.leave_one_site_out:
            return None
        if len(sites) - 1 < self._privacy.min_clients_per_aggregate:
            return LeaveOneSiteOut(status="skipped_too_few_sites")
        index = {name: i for i, name in enumerate(self._features)}
        selections: dict[str, tuple[str, ...]] = {}
        for left_out in sites:
            reduced_reliability = (
                None
                if self._reliability is None
                else [s for s in self._reliability if s.client_id != left_out]
            )
            if reduced_reliability is not None and (
                len(reduced_reliability) < self._privacy.min_clients_per_aggregate
            ):
                return LeaveOneSiteOut(status="skipped_too_few_reliability_sites")
            stats = self._with_correlation(
                pooled_statistics(
                    self._features,
                    reduced_reliability,
                    [s for s in self._relevance if s.client_id != left_out],
                ),
                left_out,
            )
            chosen, _ = select(strategy, stats)
            selections[left_out] = tuple(self._features[i] for i in chosen)
        matrix = np.zeros((len(selections), len(self._features)), dtype=bool)
        for row, picked in enumerate(selections.values()):
            matrix[row, [index[name] for name in picked]] = True
        stability = nogueira_stability(matrix)
        return LeaveOneSiteOut(
            status="computed",
            selections=selections,
            nogueira_stability=None if np.isnan(stability) else stability,
            mean_jaccard_to_full=float(
                np.mean([jaccard(full_selection, picked) for picked in selections.values()])
            ),
        )
