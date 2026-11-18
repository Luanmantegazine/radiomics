"""Centralized reference analysis - only with explicit authorization.

Pools the participants of every simulated site in one place, which a real
federation must not do without an authorization to use the data centrally.
The function therefore requires a non-empty authorization statement, records
it in its output, and only works on a simulated configuration.

What it computes, for each trained federated model's feature set:

* the **centralized model**: the same configuration fitted on the pooled
  training partitions *without* site strata (one baseline / intercept);
* its evaluation on the pooled test partitions, both **unstratified** (pairs
  across sites count; one censoring distribution) and **site-stratified** (as
  the federated evaluation);
* the stratified evaluation of the **released federated model**, which must
  equal the federated result (a check of the distributed evaluation).

Feature selection is not repeated centrally: the federated selections are used.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from client.data import LocalDataset
from client.models.cox import fit_cox
from client.models.logistic import fit_horizon_logistic
from client.models.prediction import predict_conversion
from client.splits import partition
from coordinator.outcomes import ExperimentOutcome
from evaluation.metrics import horizon_statistics, pool, summarize
from experiments.simulation import build_clients
from harmonization.combat import fit_combat
from shared.config import ExperimentConfig
from shared.models import CoxModel
from shared.newton import NewtonSettings
from shared.protocol import PartitionSpec
from shared.training import CoxSpec


class AuthorizationRequired(PermissionError):
    pass


def _harmonize(config: ExperimentConfig, train: list[LocalDataset], test: list[LocalDataset]):
    h = config.harmonization
    if h is None or h.method != "combat":
        return train, test
    names = train[0].feature_names
    targets = [i for i, n in enumerate(names) if n not in h.covariates]
    covariates = [names.index(c) for c in h.covariates]
    y = np.concatenate([d.features[:, targets] for d in train])
    x = np.concatenate([d.features[:, covariates] for d in train])
    batch = np.concatenate([np.full(d.n_participants, k) for k, d in enumerate(train)])
    _, models = fit_combat(y, batch, x, empirical_bayes=h.empirical_bayes,
                           tolerance=h.eb_tolerance)

    def apply(k: int, d: LocalDataset) -> LocalDataset:
        features = d.features.copy()
        features[:, targets] = models[k].apply(d.features[:, targets], d.features[:, covariates])
        return LocalDataset(d.participant_ids, d.feature_names, features, d.time_months,
                            d.event, d.origin)

    return ([apply(k, d) for k, d in enumerate(train)],
            [apply(k, d) for k, d in enumerate(test)])


def _metrics(model, sets: list[LocalDataset], grid, thresholds, stratified: bool):
    horizons, _ = predict_conversion(model, sets[0].feature_names, sets[0].features[:1])
    out = []
    for j, horizon in enumerate(horizons):
        parts = []
        groups = sets if stratified else [_concat(sets)]
        for d in groups:
            _, p = predict_conversion(model, d.feature_names, d.features)
            if not np.isfinite(p[:, j]).all():
                parts = None
                break
            parts.append(horizon_statistics(d.time_months, d.event, p[:, j], horizon, grid))
        if parts is None:
            out.append({"horizon_months": horizon, "available": False})
            continue
        metrics = summarize(pool(parts), thresholds)
        out.append({"available": True, **{k: metrics[k] for k in (
            "horizon_months", "n", "n_events", "brier", "scaled_brier", "auc", "c_index", "o_e",
            "calibration_error")}})
    return out


def _concat(sets: list[LocalDataset]) -> LocalDataset:
    return LocalDataset(
        tuple(p for d in sets for p in d.participant_ids), sets[0].feature_names,
        np.concatenate([d.features for d in sets]),
        np.concatenate([d.time_months for d in sets]),
        np.concatenate([d.event for d in sets]), sets[0].origin,
    )


def centralized_reference(
    config: ExperimentConfig, outcome: ExperimentOutcome, authorization: str
) -> dict:
    if not authorization or not authorization.strip():
        raise AuthorizationRequired(
            "pooling individual data needs an explicit authorization statement"
        )
    if config.simulation is None or config.training is None or config.evaluation is None:
        raise ValueError("the centralized reference needs a simulated, trained and "
                         "evaluated experiment")
    split = config.data_split
    datasets = [c._data_source.load() for c in build_clients(outcome.experiment_id, config)]
    if split is None:
        train, test = datasets, datasets
    else:
        train = [partition(d, PartitionSpec(name="train", test_fraction=split.test_fraction,
                                            seed=split.seed)) for d in datasets]
        test = [partition(d, PartitionSpec(name="test", test_fraction=split.test_fraction,
                                           seed=split.seed)) for d in datasets]
    train, test = _harmonize(config, train, test)
    pooled_train = _concat(train)
    grid = np.array(config.evaluation.grid)
    thresholds = config.evaluation.decision_thresholds
    specs = {entry.model.name: entry.model for entry in config.training.models}
    models = []
    for trained in outcome.training.models:
        if trained.status != "trained":
            continue
        released = trained.model
        names = released.feature_names
        x = pooled_train.features[:, [pooled_train.feature_names.index(n) for n in names]]
        spec = specs[trained.name]
        o = spec.optimizer
        settings = NewtonSettings(o.ridge, o.max_iterations, o.tolerance)
        horizons = (released.horizons_months if isinstance(released, CoxModel)
                    else tuple(f.horizon_months for f in released.horizons))
        if isinstance(spec, CoxSpec):
            central = fit_cox(names, x, pooled_train.time_months, pooled_train.event, settings,
                              horizons, min_events=config.privacy.min_events)
        else:
            central = fit_horizon_logistic(names, x, pooled_train.time_months,
                                           pooled_train.event, horizons, spec.censoring,
                                           settings, min_events=config.privacy.min_events)
        models.append({
            "model_name": trained.name,
            "feature_names": list(names),
            "centralized_unstratified_fit": {
                "unstratified_evaluation": _metrics(central, test, grid, thresholds, False),
                "stratified_evaluation": _metrics(central, test, grid, thresholds, True),
            },
            "released_federated_model": {
                "unstratified_evaluation": _metrics(released, test, grid, thresholds, False),
                "stratified_evaluation": _metrics(released, test, grid, thresholds, True),
            },
        })
    return {
        "authorization": authorization.strip(),
        "computed_at": datetime.now(UTC).isoformat(),
        "data_origin": config.data_origin.value,
        "note": "Individual data pooled centrally under the authorization above.",
        "models": models,
    }
