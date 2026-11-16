"""Repeated observations assembled from several local feature tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from client.reliability.icc import ReliabilityUnavailable


@dataclass(frozen=True, slots=True)
class RepeatedMeasures:
    """Features of the same participants under several conditions (local data)."""

    participant_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray  # shape (n_participants, n_conditions, n_features)


def load_feature_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"participant_id": str}).set_index("participant_id")
    if table.index.has_duplicates:
        raise ValueError(f"duplicate participants in {path}")
    return table


def assemble(tables: Mapping[str, pd.DataFrame]) -> RepeatedMeasures:
    """Align tables on the participants and features they all share.

    Only participants observed under *every* condition are kept (complete
    cases), since the ICC needs balanced data. Features must be identical
    across conditions: comparing different feature sets is a configuration
    error, not something to intersect away.
    """
    if len(tables) < 2:
        raise ReliabilityUnavailable(
            "reliability needs at least two conditions (repeated or alternative observations)"
        )
    conditions = tuple(tables)
    first = tables[conditions[0]]
    for name in conditions[1:]:
        if list(tables[name].columns) != list(first.columns):
            raise ValueError(f"condition {name!r} has a different feature schema")
    common = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    if not common:
        raise ReliabilityUnavailable("no participant was observed under every condition")
    values = np.stack(
        [tables[name].loc[common].to_numpy(dtype=float) for name in conditions], axis=1
    )
    return RepeatedMeasures(tuple(common), conditions, tuple(first.columns), values)
