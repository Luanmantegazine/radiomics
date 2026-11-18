"""Random survival forest behind the :class:`client.models.base.SurvivalModel`
interface (scikit-survival). Local only: it has no federated protocol.

Limitations: survival probabilities are step functions evaluated at the
horizons; beyond the largest training time the last value is *not* used (NaN).
Hyper-parameters are configuration, not tuned here.
"""

from __future__ import annotations

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from shared.training import RandomSurvivalForestSpec


class RandomSurvivalForestModel:
    def __init__(self, spec: RandomSurvivalForestSpec, horizons: tuple[float, ...]) -> None:
        self.name = spec.name
        self._spec, self._horizons = spec, horizons
        self._forest: RandomSurvivalForest | None = None
        self._max_time = 0.0

    def fit(self, x, time, event) -> RandomSurvivalForestModel:
        self._forest = RandomSurvivalForest(
            n_estimators=self._spec.n_estimators,
            min_samples_leaf=self._spec.min_samples_leaf,
            random_state=self._spec.seed,
            n_jobs=1,
        ).fit(np.asarray(x, dtype=float), Surv.from_arrays(event=event, time=time))
        self._max_time = float(np.max(time))
        return self

    def predict_conversion(self, x) -> np.ndarray:
        if self._forest is None:
            raise RuntimeError("model not fitted")
        functions = self._forest.predict_survival_function(np.asarray(x, dtype=float))
        out = np.full((len(functions), len(self._horizons)), np.nan)
        for j, horizon in enumerate(self._horizons):
            if horizon <= self._max_time:
                out[:, j] = [1.0 - f(horizon) for f in functions]
        return out
