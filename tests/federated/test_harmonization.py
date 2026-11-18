"""ComBat: reference implementation, federated equivalence, site-effect assessment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from client.splits import partition
from coordinator.engine import FederatedCoordinator
from coordinator.transport import InProcessTransport
from experiments.simulation import build_clients
from harmonization.assessment import event_rate_chi2, site_effect
from harmonization.combat import fit_combat
from shared.config import ExperimentConfig, load_experiment_config
from shared.protocol import PartitionSpec, TaskKind
from tests.federated.conftest import CONFIG_PATH

STAGE6 = CONFIG_PATH.parent / "stage6_harmonization_evaluation.yaml"
TRAIN = PartitionSpec(name="train", test_fraction=0.3, seed=7)


def stage6(**changes) -> ExperimentConfig:
    raw = load_experiment_config(STAGE6).model_dump(mode="json")
    raw.update(changes)
    return ExperimentConfig.model_validate(raw)


def _harmonization_only(**harmonization) -> ExperimentConfig:
    raw = load_experiment_config(STAGE6).model_dump(mode="json")
    raw["harmonization"].update(harmonization)
    raw.update(feature_selection=None, training=None, evaluation=None)
    return ExperimentConfig.model_validate(raw)


def _run(config, clients=None, capture=False):
    transport = InProcessTransport(capture=capture)
    clients = clients or build_clients("exp-h", config)
    for client in clients:
        if client._dataset is None:
            client.initialize()
        transport.attach(client)
    return FederatedCoordinator("exp-h", config, transport).run(), clients, transport


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n = [60, 45, 80]
    batch = np.repeat([1, 2, 3], n)
    age = rng.normal(70, 6, sum(n))
    y = rng.normal(size=(sum(n), 12)) + 0.05 * age[:, None]
    y += np.array([0.0, 1.5, -1.0])[batch - 1][:, None] * rng.uniform(0.5, 1.5, 12)
    y *= np.array([1.0, 1.8, 0.7])[batch - 1][:, None]
    return y, batch, age


@pytest.mark.parametrize("empirical_bayes", [True, False])
def test_combat_matches_neurocombat(empirical_bayes):
    from neuroCombat import neuroCombat

    y, batch, age = _synthetic()
    reference = neuroCombat(y.T, pd.DataFrame({"batch": batch, "age": age}), "batch",
                            continuous_cols=["age"], eb=empirical_bayes)["data"].T
    mine, _ = fit_combat(y, batch, age[:, None], empirical_bayes=empirical_bayes)
    # neuroCombat computes in float32; agreement is at that precision.
    np.testing.assert_allclose(mine, reference, atol=2e-5 * np.abs(reference).max())


def test_combat_removes_site_effects_and_keeps_covariate_effects():
    y, batch, age = _synthetic()
    adjusted, _ = fit_combat(y, batch, age[:, None])
    labels = np.unique(batch)
    def eta(data):
        n = np.array([(batch == b).sum() for b in labels])
        return site_effect(n, np.array([data[batch == b].sum(0) for b in labels]),
                           np.array([(data[batch == b] ** 2).sum(0) for b in labels]))[
            "eta_squared"]
    assert np.median(eta(y)) > 0.3 and np.max(eta(adjusted)) < 0.01
    slope = np.polyfit(age, adjusted[:, 0], 1)[0]
    assert slope == pytest.approx(0.05, abs=0.03)


def test_site_effect_equals_one_way_anova():
    y, batch, _ = _synthetic(1)
    labels = np.unique(batch)
    n = np.array([(batch == b).sum() for b in labels])
    result = site_effect(n, np.array([y[batch == b].sum(0) for b in labels]),
                         np.array([(y[batch == b] ** 2).sum(0) for b in labels]))
    grand = y.mean(0)
    between = sum(((y[batch == b].mean(0) - grand) ** 2) * (batch == b).sum() for b in labels)
    total = ((y - grand) ** 2).sum(0)
    np.testing.assert_allclose(result["eta_squared"], between / total, rtol=1e-10)
    within = total - between
    np.testing.assert_allclose(result["f_statistic"], (between / 2) / (within / (len(y) - 3)),
                               rtol=1e-10)
    assert event_rate_chi2(np.array([100, 100]), np.array([50, 50])) == 0.0
    assert event_rate_chi2(np.array([100, 100]), np.array([20, 80])) == pytest.approx(72.0)


# --------------------------------------------------------------------------- #
# Federated
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def federated():
    config = _harmonization_only()
    outcome, clients, transport = _run(config, capture=True)
    return config, outcome, clients, transport


def test_federated_combat_equals_centralized_combat(federated):
    config, outcome, clients, _ = federated
    assert outcome.completed
    assert [r.task for r in outcome.rounds] == [
        TaskKind.FEATURE_INVENTORY, TaskKind.HARMONIZATION_STATISTICS,
        TaskKind.HARMONIZATION_VARIANCE, TaskKind.HARMONIZATION_APPLY,
    ]
    datasets = [c._dataset for c in clients]
    names = datasets[0].feature_names
    targets = [i for i, n in enumerate(names) if n != "clin_age"]
    age = names.index("clin_age")
    train = [partition(d, TRAIN) for d in datasets]
    y = np.concatenate([d.features[:, targets] for d in train])
    x = np.concatenate([d.features[:, [age]] for d in train])
    batch = np.concatenate([np.full(d.n_participants, k) for k, d in enumerate(train)])
    _, models = fit_combat(y, batch, x, empirical_bayes=True, tolerance=1e-4)
    for k, (client, dataset) in enumerate(zip(clients, datasets, strict=True)):
        harmonized = client._harmonizer.apply(dataset)
        expected = models[k].apply(dataset.features[:, targets], dataset.features[:, [age]])
        np.testing.assert_allclose(harmonized.features[:, targets], expected, atol=1e-10)
        # The biological covariate is not modified.
        np.testing.assert_array_equal(harmonized.features[:, age], dataset.features[:, age])


def test_assessment_before_and_after(federated):
    _, outcome, _, _ = federated
    result = outcome.harmonization
    assert result.site_effect_before["median_eta_squared"] > 0.1
    assert result.site_effect_after["max_eta_squared"] < 0.01
    assert result.covariate_eta_squared["clin_age"] > 0.05  # sites differ clinically
    assert result.event_rate_chi2 >= 0
    assert len(result.sites) == 3 and result.follow_up_range_months[0] > 0
    before = outcome.rounds[1].aggregate
    after = outcome.rounds[3].aggregate
    assert (before.stage, after.stage) == ("before", "after")
    assert "clin_age" not in before.feature_names


def test_estimated_on_training_partitions_only(federated):
    """Changing held-out participants leaves every site's adjustment unchanged."""
    config, _, clients, _ = federated
    altered = build_clients("exp-h", config)
    for client in altered:
        client.initialize()
        d = client._dataset
        held_out = ~np.isin(d.participant_ids, partition(d, TRAIN).participant_ids)
        features = d.features.copy()
        features[held_out] += 100.0
        client._dataset = type(d)(d.participant_ids, d.feature_names, features, d.time_months,
                                  d.event, d.origin)
    _run(config, altered)
    for original, changed in zip(clients, altered, strict=True):
        np.testing.assert_allclose(original._harmonizer.model.gamma_star,
                                   changed._harmonizer.model.gamma_star, rtol=1e-12)


def test_site_adjustments_never_leave_the_site(federated):
    _, outcome, clients, transport = federated
    local = set()
    for client in clients:
        local.update(client._harmonizer.model.gamma_star.tolist())
        local.update(client._harmonizer.model.delta_star.tolist())
    wire = set()
    for record in transport.log.records:
        def walk(node):
            if isinstance(node, float):
                wire.add(node)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(json.loads(record.raw))
    assert local and not (local & wire)
    released = json.loads(outcome.model_dump_json())
    assert not any(repr(v) in json.dumps(released) for v in local)


def test_assessment_only_does_not_change_the_data():
    outcome, clients, _ = _run(_harmonization_only(method="none"))
    assert [r.task for r in outcome.rounds][-1] is TaskKind.HARMONIZATION_STATISTICS
    assert outcome.harmonization.site_effect_after is None
    assert all(c._harmonizer is None for c in clients)


def test_site_summaries_are_released_only_when_allowed():
    outcome, _, _ = _run(_harmonization_only(release_site_summaries=False))
    assert outcome.harmonization.sites == ()
    assert outcome.harmonization.follow_up_range_months is not None


def test_unknown_covariate_fails_harmonization():
    outcome, _, _ = _run(_harmonization_only(covariates=["not_a_feature"]))
    assert not outcome.completed and outcome.harmonization is None


def test_evaluation_configuration_rules():
    raw = load_experiment_config(STAGE6).model_dump(mode="json")
    for change, message in (
        ({"decision_thresholds": [0.15]}, "bin edges"),
        ({"release_site_metrics": False}, "requires release_site_metrics"),
    ):
        broken = json.loads(json.dumps(raw))
        broken["evaluation"].update(change)
        with pytest.raises(ValidationError, match=message):
            ExperimentConfig.model_validate(broken)
    broken = json.loads(json.dumps(raw))
    broken["data_split"] = None
    with pytest.raises(ValidationError, match="needs data_split"):
        ExperimentConfig.model_validate(broken)
