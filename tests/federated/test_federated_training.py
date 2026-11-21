"""Federated training: equivalence with centralized fits, partitions, isolation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from client.models.cox import breslow_cumulative_hazard, partial_likelihood
from client.models.logistic import horizon_labels, weighted_logistic
from client.node import FederatedClient
from client.splits import partition
from client.splits import test_mask as split_test_mask
from coordinator.engine import FederatedCoordinator
from coordinator.transport import Direction, InProcessTransport
from experiments.simulation import build_clients, run_simulation
from shared.config import ExperimentConfig, load_experiment_config
from shared.newton import NewtonOptimizer, NewtonSettings
from shared.protocol import PartitionSpec, TaskKind, UpdateStatus
from tests.federated.conftest import CONFIG_PATH

STAGE5 = CONFIG_PATH.parent / "stage5_federated_models.yaml"
SPLIT = PartitionSpec(name="train", test_fraction=0.3, seed=7)
SETTINGS = NewtonSettings(ridge=1.0, max_iterations=25, tolerance=1e-10)


def stage5(**changes) -> ExperimentConfig:
    raw = load_experiment_config(STAGE5).model_dump(mode="json")
    raw.update(changes)
    return ExperimentConfig.model_validate(raw)


@pytest.fixture(scope="module")
def run():
    transport = InProcessTransport(capture=True)
    config = stage5()
    return config, run_simulation("exp-s5", config, transport=transport), transport


@pytest.fixture(scope="module")
def train_sets(run):
    config, _, _ = run
    return [partition(c._data_source.load(), SPLIT) for c in build_clients("exp-s5", config)]


def _model(outcome, name):
    return next(m for m in outcome.training.models if m.name == name)


def _design(train_sets, model):
    """Each site's features, standardised with the released global centre/scale."""
    out = []
    for d in train_sets:
        x = d.features[:, [d.feature_names.index(n) for n in model.feature_names]]
        out.append((x - np.array(model.center)) / np.array(model.scale))
    return out


def _optimize(n, evaluate, penalized=None):
    optimizer = NewtonOptimizer(n, SETTINGS, penalized)
    while not optimizer.finished:
        optimizer.observe(*evaluate(optimizer.proposal))
    return optimizer


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
def test_all_models_train_and_converge(run):
    _, outcome, _ = run
    assert outcome.completed
    assert [m.status for m in outcome.training.models] == ["trained"] * 6
    phases = {r.phase.value for r in outcome.rounds}
    assert phases == {"feature_selection", "training"}
    for trained in outcome.training.models:
        fits = [trained.model.fit] if trained.model.kind == "cox" else [
            h.fit for h in trained.model.horizons]
        assert all(f.converged and f.n_clients == 3 for f in fits)
        assert all(f.evaluations <= 8 for f in fits)
    assert outcome.training_s > 0


def test_standardization_is_the_pooled_training_moments(run, train_sets):
    _, outcome, _ = run
    model = _model(outcome, "cox_reference").model
    pooled = np.concatenate([
        d.features[:, [d.feature_names.index(n) for n in model.feature_names]]
        for d in train_sets
    ])
    np.testing.assert_allclose(model.center, pooled.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(model.scale, pooled.std(axis=0, ddof=1), rtol=1e-12)


def test_federated_cox_equals_centralized_site_stratified_cox(run, train_sets):
    _, outcome, _ = run
    model = _model(outcome, "cox_selected").model
    z = np.concatenate(_design(train_sets, model))
    time = np.concatenate([d.time_months for d in train_sets])
    event = np.concatenate([d.event for d in train_sets])
    strata = np.concatenate([np.full(d.n_participants, k) for k, d in enumerate(train_sets)])
    central = _optimize(z.shape[1], lambda b: partial_likelihood(z, time, event, b, strata))
    np.testing.assert_allclose(model.coefficients, central.best, atol=1e-8)
    np.testing.assert_allclose(model.standard_errors, central.standard_errors(), rtol=1e-8)
    assert model.fit.n_samples == len(time) and model.fit.n_events == int(event.sum())
    # Released baseline: site baselines averaged with weights = site size.
    sites = [breslow_cumulative_hazard(zk, d.time_months, d.event, central.best,
                                       model.horizons_months)
             for zk, d in zip(_design(train_sets, model), train_sets, strict=True)]
    sizes = np.array([d.n_participants for d in train_sets], dtype=float)
    expected = (sizes[:, None] * np.array(sites)).sum(axis=0) / sizes.sum()
    np.testing.assert_allclose(model.baseline_cumulative_hazard, expected, rtol=1e-10)


def test_federated_logistic_equals_centralized_fits(run, train_sets):
    _, outcome, _ = run
    common = _model(outcome, "logistic_ipcw_common").model
    site = _model(outcome, "logistic_ipcw_site").model
    for h, horizon in enumerate((12.0, 24.0, 36.0)):
        labels = [horizon_labels(d.time_months, d.event, horizon, "ipcw") for d in train_sets]
        y = np.concatenate([lab[0] for lab in labels])
        w = np.concatenate([lab[1] for lab in labels])
        z = np.concatenate(_design(train_sets, common))
        design = np.column_stack([np.ones(len(z)), z])
        central = _optimize(
            design.shape[1],
            lambda b, d=design, y=y, w=w: weighted_logistic(d, y, w, b),
            np.r_[False, np.ones(z.shape[1], dtype=bool)],
        )
        fit = common.horizons[h]
        assert fit.intercept == pytest.approx(central.best[0], abs=1e-7)
        np.testing.assert_allclose(fit.coefficients, central.best[1:], atol=1e-7)

        # Site-specific intercepts: joint centralized fit with one intercept per site.
        k = len(train_sets)
        dummies = np.concatenate([
            np.tile(np.eye(k)[s], (d.n_participants, 1)) for s, d in enumerate(train_sets)
        ])
        joint_design = np.column_stack([dummies, z])
        joint = _optimize(joint_design.shape[1],
                          lambda b, d=joint_design, y=y, w=w: weighted_logistic(d, y, w, b),
                          np.r_[np.zeros(k, dtype=bool), np.ones(z.shape[1], dtype=bool)])
        np.testing.assert_allclose(site.horizons[h].coefficients, joint.best[k:], atol=1e-6)
        sizes = np.array([(lab[1] > 0).sum() for lab in labels], dtype=float)
        assert site.horizons[h].intercept == pytest.approx(
            float(np.dot(sizes, joint.best[:k]) / sizes.sum()), abs=1e-6
        )


def test_site_specific_intercepts_remove_confounding_by_site(run):
    """feat_moderate raises the hazard at every site, but its mean differs a lot
    between sites; a common intercept turns that into a spurious negative slope."""
    _, outcome, _ = run
    moderate = 1  # position of feat_moderate in the selected features
    common = _model(outcome, "logistic_ipcw_common").model.horizons[1].coefficients[moderate]
    site = _model(outcome, "logistic_ipcw_site").model.horizons[1].coefficients[moderate]
    cox = _model(outcome, "cox_selected").model.coefficients[moderate]
    assert cox > 0 and site > 0 and common < 0


# --------------------------------------------------------------------------- #
# Held-out partition
# --------------------------------------------------------------------------- #
class _Altered:
    """Data source whose held-out participants are replaced by noise."""

    def __init__(self, inner, spec):
        self._inner, self._spec, self.origin = inner, spec, inner.origin

    def load(self):
        dataset = self._inner.load()
        held_out = split_test_mask(dataset, self._spec.test_fraction, self._spec.seed)
        rng = np.random.default_rng(99)
        features = dataset.features.copy()
        features[held_out] = rng.normal(size=(held_out.sum(), features.shape[1])) * 50
        time = dataset.time_months.copy()
        time[held_out] = rng.uniform(1, 60, held_out.sum())
        return replace(dataset, features=features, time_months=time)


def test_held_out_participants_never_influence_selection_or_training(run):
    config, outcome, _ = run
    clients = build_clients("exp-s5", config)
    for client in clients:
        client._data_source = _Altered(client._data_source, SPLIT)
        client.initialize()
    transport = InProcessTransport()
    for client in clients:
        transport.attach(client)
    altered = FederatedCoordinator("exp-s5", config, transport).run()
    assert altered.feature_selection == outcome.feature_selection
    assert altered.training == outcome.training


def test_without_a_split_everyone_is_used(run):
    _, outcome, _ = run
    full = run_simulation("exp-s5", stage5(data_split=None))
    assert full.training.models[0].model.fit.n_samples == 900
    assert outcome.training.models[0].model.fit.n_samples < 900


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
def test_model_fails_cleanly_when_its_features_are_unavailable():
    from coordinator.selection.results import FederatedSelectionResult, StrategyResult
    from coordinator.training.protocol import FederatedTrainingProtocol

    training = stage5().training
    selection = FederatedSelectionResult(
        n_clients={"feature_relevance": 3}, n_features_considered=15,
        candidate_pool_size=None, jaccard_between_strategies={},
        strategies=(
            StrategyResult(name="sequential_stratified", completed=True, selected=()),
            StrategyResult(name="relevance_only", completed=False, failure_reason="x"),
        ),
    )
    protocol = FederatedTrainingProtocol(training, selection)
    request = protocol.next_request()  # the explicit-feature model is the first runnable one
    assert request.task is TaskKind.FEATURE_SUMMARY
    assert request.feature_names == ("feat_strong", "feat_moderate")
    results = {m.name: m for m in protocol._results}
    assert results["cox_selected"].failure_reason == "no_feature_selected"
    assert results["cox_relevance_only"].failure_reason == "feature_selection_unavailable"
    assert all(m.status == "failed" and m.model is None for m in results.values())


class _DropOut:
    """A site that stops contributing after a few model rounds."""

    def __init__(self, inner: FederatedClient, after: int) -> None:
        self.client_id, self._inner, self._after, self._seen = inner.client_id, inner, after, 0

    def build_registration(self):
        return self._inner.build_registration()

    def handle_registration_ack(self, raw):
        self._inner.handle_registration_ack(raw)

    def handle_round(self, raw):
        if json.loads(raw)["task"] == "model_update":
            self._seen += 1
        if self._seen > self._after:  # from then on the site answers nothing usefully
            return self._inner.handle_round(raw.replace(b'"experiment_id":"exp-s5"',
                                                        b'"experiment_id":"other"'))
        return self._inner.handle_round(raw)


def test_changing_site_set_fails_the_model_instead_of_mixing_objectives():
    config = stage5()
    clients = build_clients("exp-s5", config)
    for client in clients:
        client.initialize()
    transport = InProcessTransport()
    for client in [_DropOut(clients[0], after=2), *clients[1:]]:
        transport.attach(client)
    outcome = FederatedCoordinator("exp-s5", config, transport).run()
    first = outcome.training.models[0]
    assert first.status == "failed"
    assert "set of contributing sites changed" in first.failure_reason
    # Later models start with the two remaining sites and keep them throughout.
    later = _model(outcome, "cox_reference")
    assert later.status == "trained" and later.model.fit.n_clients == 2


def test_too_few_events_refuses_model_rounds():
    config = stage5(privacy={"min_cohort_size": 10, "min_clients_per_aggregate": 2,
                             "min_events": 5}, feature_selection=None,
                    training={"horizons_months": [12, 24, 36], "models": [
                        {"model": {"kind": "horizon_logistic", "name": "short",
                                   "censoring": "ipcw", "intercept": "common",
                                   "optimizer": {"ridge": 1.0, "max_iterations": 25,
                                                 "tolerance": 1e-10}},
                         "features": {"names": ["feat_strong"]}}]})
    raw = config.model_dump(mode="json")
    raw["training"]["horizons_months"] = [0.05]  # nobody converts that early
    outcome = run_simulation("exp-s5", ExperimentConfig.model_validate(raw))
    update = next(r for r in outcome.rounds if r.task is TaskKind.MODEL_UPDATE)
    assert {c.reason_code.value for c in update.clients} == {"too_few_events"}
    assert outcome.training.models[0].status == "failed"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"training": {"horizons_months": [24, 12], "models": []}}, "increasing|at least 1"),
        ({"training": {"horizons_months": [12], "models": [
            {"model": {"kind": "random_survival_forest", "name": "rsf", "n_estimators": 10,
                       "min_samples_leaf": 5, "seed": 1},
             "features": {"names": ["feat_strong"]}}]}}, "no federated protocol"),
        ({"training": {"horizons_months": [12], "models": [
            {"model": {"kind": "cox", "name": "c", "optimizer": {
                "ridge": 1, "max_iterations": 5, "tolerance": 1e-8}},
             "features": {"from_strategy": "missing"}}]}}, "unknown strategy"),
        ({"training": {"horizons_months": [12], "models": [
            {"model": {"kind": "horizon_logistic", "name": "l", "censoring": "ipcw",
                       "optimizer": {"ridge": 1, "max_iterations": 5, "tolerance": 1e-8}},
             "features": {"names": ["feat_strong"]}}]}}, "intercept"),
    ],
)
def test_training_configuration_rules(changes, message):
    with pytest.raises(ValidationError, match=message):
        stage5(**changes)


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #
def _numbers(node):
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, int | float):
        yield float(node)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _numbers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _numbers(value)


def test_no_individual_data_on_the_wire(run):
    config, _, transport = run
    datasets = [c._data_source.load() for c in build_clients("exp-s5", config)]
    wire = b"\n".join(r.raw for r in transport.log.records).decode()
    assert not any(pid in wire for d in datasets for pid in d.participant_ids)
    individual = np.concatenate([np.r_[d.features.ravel(), d.time_months] for d in datasets])
    sent = np.array([n for r in transport.log.records
                     if r.direction is Direction.TO_COORDINATOR
                     for n in _numbers(json.loads(r.raw))])
    assert np.abs(sent[:, None] - individual[None, :]).min() > 1e-9


def test_per_site_model_quantities_are_never_released(run):
    _, outcome, transport = run
    per_site = set()
    for record in transport.log.records:
        payload = json.loads(record.raw).get("payload") or {}
        if payload.get("kind") == "model_update":
            per_site.update(payload["gradient"] + payload["information_upper"])
            per_site.add(payload["loglik"])
        elif payload.get("kind") == "baseline_hazard":
            per_site.update(v for v in payload["cumulative_hazard"] if v is not None)
        elif payload.get("kind") == "site_intercept":
            per_site.add(payload["intercept"])
    per_site.discard(0.0)
    released = set(_numbers(json.loads(outcome.model_dump_json())))
    assert per_site and not (per_site & released)
    to_clients = {n for r in transport.log.records if r.direction is Direction.TO_CLIENT
                  for n in _numbers(json.loads(r.raw))}
    assert not (per_site & to_clients)


def test_clients_receive_only_global_parameters(run):
    _, outcome, transport = run
    released = set(_numbers(json.loads(outcome.model_dump_json())))
    for record in transport.log.records:
        message = json.loads(record.raw)
        if record.direction is Direction.TO_CLIENT and message.get("model"):
            assert set(message["model"]) == {
                "kind", "center", "scale", "coefficients", "horizon_months", "censoring",
                "intercept", "horizons_months",
            }
            # Every coefficient sent is a global iterate, recorded in the round aggregates.
            assert set(message["model"]["coefficients"]) <= released | {0.0}
    assert all(c.status is UpdateStatus.COMPLETED for r in outcome.rounds for c in r.clients)
