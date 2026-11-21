from __future__ import annotations

import numpy as np
import pytest

from coordinator.aggregation import (
    AggregationError,
    PooledMoments,
    SchemaIntersection,
    get_strategy,
)
from shared.protocol import PROTOCOL_VERSION, ClientUpdate


def _update(client_id: str, task: str, payload: dict) -> ClientUpdate:
    return ClientUpdate.model_validate(
        {
            "protocol_version": PROTOCOL_VERSION, "experiment_id": "e", "round_id": "e-r001",
            "client_id": client_id, "task": task, "status": "completed", "payload": payload,
        }
    )


def test_pooled_moments_equal_centralized_statistics():
    rng = np.random.default_rng(0)
    parts = [rng.normal(loc, 1 + loc, size=(n, 3)) for loc, n in [(0, 30), (2, 55), (-1, 12)]]
    updates = [
        _update(
            f"client-{i:02d}",
            "feature_summary",
            {
                "kind": "feature_summary", "feature_names": ["a", "b", "c"], "n_samples": len(x),
                "mean": x.mean(axis=0).tolist(), "variance": x.var(axis=0, ddof=1).tolist(),
            },
        )
        for i, x in enumerate(parts)
    ]
    pooled = PooledMoments().aggregate(updates)
    everything = np.concatenate(parts)
    assert pooled.n_samples == len(everything)
    np.testing.assert_allclose(pooled.mean, everything.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(pooled.variance, everything.var(axis=0, ddof=1), rtol=1e-12)


def test_schema_intersection_keeps_only_common_features():
    updates = [
        _update("client-01", "feature_inventory",
                {"kind": "feature_inventory", "feature_names": ["a", "b", "c"]}),
        _update("client-02", "feature_inventory",
                {"kind": "feature_inventory", "feature_names": ["c", "a"]}),
    ]
    assert SchemaIntersection().aggregate(updates).feature_names == ("a", "c")


def test_strategies_are_resolved_by_name():
    assert get_strategy("pooled_moments").name == "pooled_moments"
    with pytest.raises(AggregationError, match="unknown aggregation strategy"):
        get_strategy("fedavg")
