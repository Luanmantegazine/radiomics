"""Local computations. A task is the only place where individual-level data is
turned into an authorized, aggregate payload.

Stage 1 ships two infrastructure tasks. Scientific tasks (reliability,
prognostic relevance, redundancy, local training) are added in later stages by
registering new :class:`LocalTask` implementations.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from client.data import LocalDataset
from shared.privacy import DisclosureViolation
from shared.protocol import (
    FeatureCovariancePayload,
    FeatureInventoryPayload,
    FeatureRelevancePayload,
    FeatureReliabilityPayload,
    FeatureSummaryPayload,
    ReasonCode,
    RoundConfig,
    TaskKind,
    WireModel,
)


class TaskRefused(Exception):
    """The task cannot be run locally; reported to the coordinator as a code."""

    def __init__(self, code: ReasonCode) -> None:
        super().__init__(code.value)
        self.code = code


class LocalTask(Protocol):
    kind: TaskKind

    def run(self, dataset: LocalDataset, config: RoundConfig) -> WireModel: ...


class FeatureInventoryTask:
    """Report which features exist locally (names only)."""

    kind = TaskKind.FEATURE_INVENTORY

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureInventoryPayload:
        return FeatureInventoryPayload(feature_names=dataset.feature_names)


class FeatureSummaryTask:
    """Per-feature count, mean and unbiased variance over the local cohort.

    Assumes complete data (no missing values); refuses when the cohort is
    smaller than the policy threshold or a requested feature is unavailable.
    """

    kind = TaskKind.FEATURE_SUMMARY

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureSummaryPayload:
        if not config.feature_names:
            raise TaskRefused(ReasonCode.MISSING_FEATURES)
        if dataset.n_participants < config.privacy.min_cohort_size:
            raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "local cohort below threshold")
        index = {name: i for i, name in enumerate(dataset.feature_names)}
        if any(name not in index for name in config.feature_names):
            raise TaskRefused(ReasonCode.MISSING_FEATURES)
        columns = dataset.features[:, [index[name] for name in config.feature_names]]
        return FeatureSummaryPayload(
            feature_names=config.feature_names,
            n_samples=dataset.n_participants,
            mean=tuple(np.mean(columns, axis=0).tolist()),
            variance=tuple(np.var(columns, axis=0, ddof=1).tolist()),
        )


def _columns(dataset: LocalDataset, config: RoundConfig) -> np.ndarray:
    """Columns of the requested (agreed) features, after the cohort-size rule."""
    if not config.feature_names:
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    if dataset.n_participants < config.privacy.min_cohort_size:
        raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "local cohort below threshold")
    index = {name: i for i, name in enumerate(dataset.feature_names)}
    if any(name not in index for name in config.feature_names):
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    return dataset.features[:, [index[name] for name in config.feature_names]]


class FeatureReliabilityTask:
    """Report the site's reliability summary for the agreed features.

    Refuses when the site has no reliability study: reliability is never
    reported without repeated or alternative observations behind it.
    """

    kind = TaskKind.FEATURE_RELIABILITY

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureReliabilityPayload:
        if dataset.reliability is None:
            raise TaskRefused(ReasonCode.RELIABILITY_UNAVAILABLE)
        column = config.params.get("reliability_column")
        table = dataset.reliability.values.get(str(column))
        if table is None or not config.feature_names:
            raise TaskRefused(ReasonCode.RELIABILITY_UNAVAILABLE)
        if dataset.reliability.n_subjects < config.privacy.min_cohort_size:
            raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "reliability study too small")
        values = []
        for name in config.feature_names:
            value = float(table.get(name, float("nan")))
            values.append(value if np.isfinite(value) else None)
        return FeatureReliabilityPayload(
            feature_names=config.feature_names,
            column=column,
            n_subjects=dataset.reliability.n_subjects,
            values=tuple(values),
        )


class FeatureRelevanceTask:
    """Per-feature Cox score U and information I on the site's own risk sets.

    Uses follow-up time and the event indicator: censored participants count
    while at risk and are never treated as non-converters.
    """

    kind = TaskKind.FEATURE_RELEVANCE

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureRelevancePayload:
        from client.relevance import cox_score_components  # needs the science extra

        columns = _columns(dataset, config)
        n_events = int(dataset.event.sum())
        if n_events < config.privacy.min_events:
            raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few local events")
        score, information = cox_score_components(columns, dataset.time_months, dataset.event)
        return FeatureRelevancePayload(
            feature_names=config.feature_names,
            n_samples=dataset.n_participants,
            n_events=n_events,
            score=tuple(score.tolist()),
            information=tuple(information.tolist()),
        )


class FeatureCovarianceTask:
    """Within-site scatter matrix of the candidate features (no outcome used).

    With ``rank_transform`` each column is replaced by its within-site ranks
    (average ranks for ties), which makes the pooled result a within-site
    Spearman correlation.
    """

    kind = TaskKind.FEATURE_COVARIANCE

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureCovariancePayload:
        columns = _columns(dataset, config)
        rank = bool(config.params.get("rank_transform", False))
        if rank:
            from scipy.stats import rankdata  # needs the science extra

            columns = rankdata(columns, axis=0)
        centred = columns - columns.mean(axis=0)
        scatter = centred.T @ centred
        upper = scatter[np.triu_indices(scatter.shape[0])]
        return FeatureCovariancePayload(
            feature_names=config.feature_names,
            n_samples=dataset.n_participants,
            rank_transformed=rank,
            scatter_upper=tuple(upper.tolist()),
        )


def default_tasks() -> dict[TaskKind, LocalTask]:
    tasks: list[LocalTask] = [
        FeatureInventoryTask(),
        FeatureSummaryTask(),
        FeatureReliabilityTask(),
        FeatureRelevanceTask(),
        FeatureCovarianceTask(),
    ]
    return {task.kind: task for task in tasks}
