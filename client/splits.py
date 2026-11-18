"""Deterministic train/test split of a site's participants.

Each participant's partition depends only on (seed, participant id, event
status): the split does not change with row order, and stratifying by event
keeps the event rate similar in both partitions. Within each event stratum,
participants are ordered by SHA-256 of ``seed|participant_id`` and the first
``round(test_fraction * n_stratum)`` go to the test partition.

Limitation: stratifying on the event indicator uses the outcome to build the
split (standard practice; it does not leak information into the models).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from client.data import LocalDataset
from shared.protocol import PartitionSpec


def _key(seed: int, participant_id: str) -> str:
    return hashlib.sha256(f"{seed}|{participant_id}".encode()).hexdigest()


def test_mask(dataset: LocalDataset, test_fraction: float, seed: int) -> np.ndarray:
    mask = np.zeros(dataset.n_participants, dtype=bool)
    for stratum in (True, False):
        rows = np.flatnonzero(dataset.event == stratum)
        order = sorted(rows, key=lambda i: _key(seed, dataset.participant_ids[i]))
        mask[order[: int(round(test_fraction * len(rows)))]] = True
    return mask


def partition(dataset: LocalDataset, spec: PartitionSpec | None) -> LocalDataset:
    """Rows of ``dataset`` in the requested partition (all rows when ``spec`` is None)."""
    if spec is None:
        return dataset
    test = test_mask(dataset, spec.test_fraction, spec.seed)
    rows = test if spec.name == "test" else ~test
    return replace(
        dataset,
        participant_ids=tuple(np.array(dataset.participant_ids, dtype=object)[rows]),
        features=dataset.features[rows],
        time_months=dataset.time_months[rows],
        event=dataset.event[rows],
    )
