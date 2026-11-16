"""Aggregation strategies and their registry.

A strategy is selected *by name* in the experiment configuration, so algorithms
can be swapped without touching the coordinator engine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from shared.protocol import (
    ClientUpdate,
    FeatureInventoryAggregate,
    FeatureInventoryPayload,
    FeatureSummaryAggregate,
    FeatureSummaryPayload,
    TaskKind,
    WireModel,
)


class AggregationError(Exception):
    """The updates cannot be combined by this strategy."""


class AggregationStrategy(Protocol):
    name: str
    task: TaskKind

    def aggregate(self, updates: Sequence[ClientUpdate]) -> WireModel:
        """Combine *completed* updates of one round into a global aggregate."""
        ...


class SchemaIntersection:
    """Features available at every site, in the order of the first site."""

    name = "schema_intersection"
    task = TaskKind.FEATURE_INVENTORY

    def aggregate(self, updates: Sequence[ClientUpdate]) -> FeatureInventoryAggregate:
        payloads = [u.payload for u in updates if isinstance(u.payload, FeatureInventoryPayload)]
        if len(payloads) != len(updates) or not payloads:
            raise AggregationError("schema_intersection needs feature_inventory payloads")
        common = set(payloads[0].feature_names).intersection(
            *(p.feature_names for p in payloads[1:])
        )
        ordered = tuple(name for name in payloads[0].feature_names if name in common)
        return FeatureInventoryAggregate(n_clients=len(payloads), feature_names=ordered)


class PooledMoments:
    """Exact pooled mean and unbiased variance from per-site (n, mean, variance).

    Equals the statistics of the concatenated data (up to floating point), so
    the federated result can be verified against a centralized computation.
    Assumes all sites report the same features in the same order.
    """

    name = "pooled_moments"
    task = TaskKind.FEATURE_SUMMARY

    def aggregate(self, updates: Sequence[ClientUpdate]) -> FeatureSummaryAggregate:
        payloads = [u.payload for u in updates if isinstance(u.payload, FeatureSummaryPayload)]
        if len(payloads) != len(updates) or not payloads:
            raise AggregationError("pooled_moments needs feature_summary payloads")
        names = payloads[0].feature_names
        if any(p.feature_names != names for p in payloads):
            raise AggregationError("sites reported different feature schemas")
        n = np.array([p.n_samples for p in payloads], dtype=float)[:, None]
        means = np.array([p.mean for p in payloads])
        variances = np.array([p.variance for p in payloads])
        total = n.sum()
        pooled_mean = (n * means).sum(axis=0) / total
        within = ((n - 1) * variances).sum(axis=0)
        between = (n * (means - pooled_mean) ** 2).sum(axis=0)
        return FeatureSummaryAggregate(
            n_clients=len(payloads),
            n_samples=int(total),
            feature_names=names,
            mean=tuple(pooled_mean.tolist()),
            variance=tuple(((within + between) / (total - 1)).tolist()),
        )


_REGISTRY: dict[str, Callable[[], AggregationStrategy]] = {}


def register_strategy(factory: Callable[[], AggregationStrategy]) -> None:
    _REGISTRY[factory().name] = factory


def get_strategy(name: str) -> AggregationStrategy:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise AggregationError(
            f"unknown aggregation strategy {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


register_strategy(SchemaIntersection)
register_strategy(PooledMoments)
