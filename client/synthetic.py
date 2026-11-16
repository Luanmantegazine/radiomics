"""Synthetic local data.

FOR INFRASTRUCTURE AND COMMUNICATION TESTS ONLY. The values are random noise
with no relation to MRI, radiomics or dementia; nothing computed from them may
be presented as a scientific result. Results derived from this source are
tagged ``data_origin=synthetic`` end to end.
"""

from __future__ import annotations

import numpy as np

from client.data import LocalDataset
from shared.config import SyntheticDataSourceSpec
from shared.protocol import DataOrigin


class SyntheticDataSource:
    """Seeded random dataset with the same *shape* as real prognostic data."""

    origin = DataOrigin.SYNTHETIC

    def __init__(self, spec: SyntheticDataSourceSpec, *, client_id: str, seed: int) -> None:
        self._spec = spec
        self._client_id = client_id
        self._seed = seed

    def load(self) -> LocalDataset:
        rng = np.random.default_rng(self._seed)
        n, p = self._spec.n_participants, len(self._spec.feature_names)
        location = rng.normal(0.0, 1.0, size=p)  # site-specific offset
        return LocalDataset(
            participant_ids=tuple(f"SYN-{self._client_id}-{i:05d}" for i in range(n)),
            feature_names=self._spec.feature_names,
            features=rng.normal(location, 1.0, size=(n, p)),
            time_months=rng.uniform(1.0, 60.0, size=n),
            event=rng.random(n) < 0.4,
            origin=DataOrigin.SYNTHETIC,
        )
