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
    FeatureInventoryPayload,
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


def default_tasks() -> dict[TaskKind, LocalTask]:
    tasks: list[LocalTask] = [FeatureInventoryTask(), FeatureSummaryTask()]
    return {task.kind: task for task in tasks}
