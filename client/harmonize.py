"""Site-side harmonization state: the site's own ComBat adjustment.

The site's location/scale parameters (gamma*, delta*) never leave the site; the
harmonizer is applied to every later task (selection, training, evaluation),
to the training and held-out partitions alike.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from client.data import LocalDataset
from harmonization.combat import SiteCombat


def columns(dataset: LocalDataset, names: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(dataset.feature_names)}
    missing = [n for n in names if n not in index]
    if missing:
        raise KeyError(missing)
    return dataset.features[:, [index[n] for n in names]]


@dataclass(frozen=True, slots=True)
class SiteHarmonizer:
    feature_names: tuple[str, ...]
    covariates: tuple[str, ...]
    model: SiteCombat

    def apply(self, dataset: LocalDataset) -> LocalDataset:
        index = {name: i for i, name in enumerate(dataset.feature_names)}
        features = dataset.features.copy()
        features[:, [index[n] for n in self.feature_names]] = self.model.apply(
            columns(dataset, self.feature_names), columns(dataset, self.covariates)
        )
        return replace(dataset, features=features)


@dataclass(frozen=True, slots=True)
class LocalStateUpdate:
    """Returned by a task that changes the site's local state, with its payload."""

    payload: object
    harmonizer: SiteHarmonizer
