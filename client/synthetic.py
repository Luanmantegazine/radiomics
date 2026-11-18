"""Synthetic local data.

FOR INFRASTRUCTURE AND COMMUNICATION TESTS ONLY. The values are random noise
with no relation to MRI, radiomics or dementia; nothing computed from them may
be presented as a scientific result. Results derived from this source are
tagged ``data_origin=synthetic`` end to end.
"""

from __future__ import annotations

import numpy as np

from client.data import LocalDataset, ReliabilityEstimates
from client.reliability.icc import IccForm, icc
from shared.config import SyntheticDataSourceSpec
from shared.protocol import DataOrigin


class SyntheticDataSource:
    """Seeded random dataset with the same *shape* as real prognostic data.

    Conversion times are exponential with hazard ``BASE_HAZARD * exp(x . beta)``
    (beta from ``log_hazard_effects``; zero by default) and are right-censored
    by an independent uniform follow-up. When ``measurement_noise_sd`` is set,
    a second observation ``x + noise`` is simulated and reliability is
    *estimated* from the two observations (ICC(2,1)), never stated.
    """

    BASE_HAZARD = 0.02  # per month
    MAX_FOLLOW_UP = 60.0

    origin = DataOrigin.SYNTHETIC

    def __init__(self, spec: SyntheticDataSourceSpec, *, client_id: str, seed: int) -> None:
        self._spec = spec
        self._client_id = client_id
        self._seed = seed

    def load(self) -> LocalDataset:
        rng = np.random.default_rng(self._seed)
        n, p = self._spec.n_participants, len(self._spec.feature_names)
        names = self._spec.feature_names
        location = rng.normal(0.0, 1.0, size=p)  # site-specific offset
        features = rng.normal(location, 1.0, size=(n, p))
        column = {name: i for i, name in enumerate(names)}
        for target, derived in self._spec.derived_features.items():
            features[:, column[target]] = (
                features[:, column[derived.source]]
                - location[column[derived.source]]
                + location[column[target]]
                + rng.normal(0.0, derived.noise_sd, size=n)
            )
        beta = np.array([self._spec.log_hazard_effects.get(name, 0.0) for name in names])
        hazard = self.BASE_HAZARD * np.exp((features - location) @ beta)
        event_time = rng.exponential(1.0 / hazard)
        follow_up = rng.uniform(1.0, self.MAX_FOLLOW_UP, size=n)
        reliability = None
        if self._spec.measurement_noise_sd is not None:
            sd = np.array([self._spec.measurement_noise_sd.get(name, 0.0) for name in names])
            repeat = features + rng.normal(size=(n, p)) * sd
            result = icc(np.stack([features, repeat], axis=1), IccForm.ICC2)
            reliability = ReliabilityEstimates(
                values={
                    "min_icc": dict(zip(names, result.estimate.tolist(), strict=True)),
                    "min_ci_low": dict(zip(names, result.ci_low.tolist(), strict=True)),
                },
                n_subjects=n,
            )
        return LocalDataset(
            participant_ids=tuple(f"SYN-{self._client_id}-{i:05d}" for i in range(n)),
            feature_names=names,
            features=features,
            time_months=np.minimum(event_time, follow_up),
            event=event_time <= follow_up,
            origin=DataOrigin.SYNTHETIC,
            reliability=reliability,
        )
