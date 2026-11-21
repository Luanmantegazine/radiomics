"""Federated feature selection end to end (synthetic sites with a planted structure)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pytest
from pydantic import ValidationError

from client.node import FederatedClient
from coordinator.transport import Direction, InProcessTransport
from experiments.simulation import build_clients, run_simulation
from shared.config import ExperimentConfig, load_experiment_config
from shared.protocol import (
    FeatureCovariancePayload,
    FeatureRelevancePayload,
    FeatureReliabilityPayload,
    TaskKind,
    UpdateStatus,
)
from tests.federated.conftest import CONFIG_PATH

STAGE4 = CONFIG_PATH.parent / "stage4_federated_selection.yaml"


def stage4(**changes) -> ExperimentConfig:
    raw = load_experiment_config(STAGE4).model_dump(mode="json")
    for key, value in changes.items():
        if key.startswith("selection__"):
            raw["feature_selection"][key.removeprefix("selection__")] = value
        else:
            raw[key] = value
    return ExperimentConfig.model_validate(raw)


def _site(config: ExperimentConfig, index: int, **source_changes) -> dict:
    raw = config.model_dump(mode="json")["simulation"]["clients"][index]
    raw["data_source"].update(source_changes)
    return raw


@pytest.fixture(scope="module")
def run():
    transport = InProcessTransport(capture=True)
    config = stage4()
    outcome = run_simulation("exp-s4", config, transport=transport)
    return config, outcome, transport


def _strategy(outcome, name):
    return next(s for s in outcome.feature_selection.strategies if s.name == name)


def _selected(outcome, name) -> list[str]:
    return [f.feature for f in _strategy(outcome, name).selected]


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
def test_rounds_follow_the_protocol(run):
    _, outcome, _ = run
    assert outcome.completed
    assert [r.task for r in outcome.rounds] == [
        TaskKind.FEATURE_INVENTORY, TaskKind.FEATURE_RELIABILITY,
        TaskKind.FEATURE_RELEVANCE, TaskKind.FEATURE_COVARIANCE,
    ]
    assert all(r.phase.value == "feature_selection" for r in outcome.rounds)
    assert all(len(r.clients) == 3 for r in outcome.rounds)
    relevance = outcome.rounds[2].aggregate
    assert relevance.n_samples == 900 and relevance.n_clients == 3
    covariance = outcome.rounds[3].aggregate
    pool = outcome.feature_selection.candidate_pool_size
    assert len(covariance.feature_names) == pool and len(covariance.correlation_upper) == (
        pool * (pool + 1) // 2
    )
    assert outcome.selection_s > 0


def test_each_dimension_acts_as_designed(run):
    _, outcome, _ = run
    selected = _selected(outcome, "sequential_stratified")
    assert {"feat_strong", "feat_moderate"} <= set(selected)
    assert "feat_noisy" not in selected  # unreliable (reliability)
    assert "feat_mixed" not in selected  # opposite effects across sites (agreement)
    assert "feat_copy" not in selected  # near-copy of feat_strong (redundancy)
    excluded = _strategy(outcome, "sequential_stratified").excluded_counts
    assert excluded["below_reliability_threshold"] == 1
    assert excluded["below_agreement_threshold"] >= 1
    # Without the three filters the same data yields exactly the planted traps.
    assert _selected(outcome, "relevance_only") == ["feat_strong", "feat_copy", "feat_noisy"]


def test_released_scores(run):
    _, outcome, _ = run
    top = _strategy(outcome, "sequential_stratified").selected[0]
    assert top.rank == 1 and top.feature == "feat_strong"
    assert top.reliability > 0.9 and top.agreement > 0.5
    assert top.relevance_z > 5 and top.relevance_p < 1e-6
    mean_rank = _strategy(outcome, "sequential_mean_rank").selected[0]
    assert mean_rank.relevance_z is None and 0 < mean_rank.ranking_score <= 1


def test_strategies_are_compared_without_extra_communication(run):
    config, outcome, _ = run
    names = [s.name for s in config.feature_selection.strategies]
    assert [s.name for s in outcome.feature_selection.strategies] == names
    assert len(outcome.feature_selection.jaccard_between_strategies) == 15
    assert outcome.feature_selection.jaccard_between_strategies[
        "sequential_stratified|relevance_only"
    ] < 1
    assert len(outcome.rounds) == 4  # one set of rounds for six strategies


def test_leave_one_site_out_stability(run):
    _, outcome, _ = run
    loso = _strategy(outcome, "sequential_stratified").leave_one_site_out
    assert loso.status == "computed"
    assert set(loso.selections) == {"client-01", "client-02", "client-03"}
    assert all(len(v) <= 3 for v in loso.selections.values())
    assert -1 <= loso.nogueira_stability <= 1 and 0 <= loso.mean_jaccard_to_full <= 1
    assert all("feat_strong" in v for v in loso.selections.values())


def test_selection_is_reproducible(run):
    config, outcome, _ = run
    again = run_simulation("exp-s4", config)
    assert again.feature_selection == outcome.feature_selection
    assert [r.aggregate for r in again.rounds] == [r.aggregate for r in outcome.rounds]


def test_optional_rounds_are_skipped():
    strategy = {
        "name": "plain", "reliability": None, "relevance": {"aggregation": "stratified_score"},
        "agreement": None, "ranking": {"kind": "sequential"}, "max_abs_correlation": None,
        "n_features": 2,
    }
    config = stage4(selection__reliability_column=None, selection__redundancy=None,
                    selection__strategies=[strategy])
    outcome = run_simulation("exp-s4", config)
    assert outcome.completed
    assert [r.task for r in outcome.rounds] == [
        TaskKind.FEATURE_INVENTORY, TaskKind.FEATURE_RELEVANCE,
    ]
    assert outcome.feature_selection.n_clients == {"feature_relevance": 3}
    assert outcome.feature_selection.candidate_pool_size is None


def test_selection_after_a_fixed_inventory_round_reuses_the_schema():
    config = stage4(rounds=[{"phase": "infrastructure", "task": "feature_inventory",
                             "aggregator": "schema_intersection"}])
    outcome = run_simulation("exp-s4", config)
    assert [r.task.value for r in outcome.rounds][:2] == ["feature_inventory",
                                                          "feature_reliability"]


# --------------------------------------------------------------------------- #
# Sites that cannot or may not contribute
# --------------------------------------------------------------------------- #
def test_site_without_reliability_study_refuses_and_others_are_used():
    config = stage4()
    clients = [_site(config, 0, measurement_noise_sd=None), _site(config, 1), _site(config, 2)]
    outcome = run_simulation("exp-s4", stage4(simulation={"clients": clients}))
    reliability_round = outcome.rounds[1]
    first = reliability_round.clients[0]
    assert first.status is UpdateStatus.REFUSED
    assert first.reason_code.value == "reliability_unavailable"
    assert reliability_round.aggregate.n_clients == 2
    assert outcome.feature_selection.n_clients["feature_reliability"] == 2
    assert outcome.feature_selection.n_clients["feature_relevance"] == 3
    loso = _strategy(outcome, "sequential_stratified").leave_one_site_out
    assert loso.status == "skipped_too_few_reliability_sites"


def test_reliability_is_never_invented_when_too_few_sites_estimate_it():
    config = stage4()
    clients = [_site(config, 0), _site(config, 1, measurement_noise_sd=None),
               _site(config, 2, measurement_noise_sd=None)]
    outcome = run_simulation("exp-s4", stage4(simulation={"clients": clients}))
    assert not outcome.completed
    assert outcome.rounds[-1].task is TaskKind.FEATURE_RELIABILITY
    assert outcome.rounds[-1].failure_reason == "too_few_valid_updates"
    assert outcome.feature_selection is None


def test_site_with_too_few_events_does_not_report_outcome_statistics():
    config = stage4(privacy={"min_cohort_size": 10, "min_clients_per_aggregate": 2,
                             "min_events": 400})
    outcome = run_simulation("exp-s4", config)
    relevance = next(r for r in outcome.rounds if r.task is TaskKind.FEATURE_RELEVANCE)
    assert {c.reason_code.value for c in relevance.clients} == {"too_few_events"}
    assert not outcome.completed


def test_loso_needs_enough_sites():
    config = stage4()
    two = stage4(expected_clients=["client-01", "client-02"],
                 simulation={"clients": [_site(config, 0), _site(config, 1)]})
    outcome = run_simulation("exp-s4", two)
    assert outcome.completed
    assert _strategy(outcome, "sequential_stratified").leave_one_site_out.status == (
        "skipped_too_few_sites"
    )


# --------------------------------------------------------------------------- #
# Configuration rules
# --------------------------------------------------------------------------- #
def _strategies(**changes):
    raw = load_experiment_config(STAGE4).model_dump(mode="json")["feature_selection"]
    raw["strategies"][0].update(changes)
    return raw["strategies"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selection__reliability_column": None}, "uses reliability but reliability_column"),
        ({"selection__redundancy": None}, "redundancy round is disabled"),
        ({"selection__redundancy": {"correlation": "spearman", "max_candidates": 50}},
         "must not exceed 200"),
        ({"selection__strategies": _strategies(name="relevance_only")}, "unique"),
        ({"selection__strategies": _strategies(
            agreement=None, ranking={"kind": "weighted",
                                     "weights": {"reliability": 0, "relevance": 1,
                                                 "agreement": 1}})},
         "weighting agreement needs an agreement criterion"),
        ({"selection__strategies": _strategies(
            ranking={"kind": "weighted",
                     "weights": {"reliability": 0, "relevance": 0, "agreement": 0}})},
         "at least one weight"),
        ({"selection__strategies": _strategies(n_features=25)}, "n_features > max_candidates"),
    ],
)
def test_configuration_rules(changes, message):
    with pytest.raises(ValidationError, match=message):
        stage4(**changes)


def test_no_threshold_has_a_default():
    raw = load_experiment_config(STAGE4).model_dump(mode="json")
    for field in ("max_abs_correlation", "n_features", "reliability", "agreement"):
        broken = json.loads(json.dumps(raw))
        del broken["feature_selection"]["strategies"][0][field]
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(broken)


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #
def _numbers(node) -> Iterator[float]:
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


def _datasets(config):
    clients = build_clients("exp-s4", config)
    return [c._data_source.load() for c in clients]


def test_no_individual_data_crosses_the_boundary(run):
    config, _, transport = run
    datasets = _datasets(config)
    wire = b"\n".join(r.raw for r in transport.log.records).decode()
    for dataset in datasets:
        assert not any(pid in wire for pid in dataset.participant_ids)
    individual = np.concatenate(
        [np.concatenate([d.features.ravel(), d.time_months]) for d in datasets]
    )
    sent = np.array([
        n for r in transport.log.records if r.direction is Direction.TO_COORDINATOR
        for n in _numbers(json.loads(r.raw))
    ])
    assert np.abs(sent[:, None] - individual[None, :]).min() > 1e-9


def test_messages_are_indexed_by_feature_never_by_participant(run):
    _, outcome, transport = run
    q = outcome.feature_selection.candidate_pool_size
    allowed = {15, q, q * (q + 1) // 2}
    for record in transport.log.records:
        message = json.loads(record.raw)
        payload = message.get("payload") or {}
        for key, value in payload.items():
            if isinstance(value, list):
                assert len(value) in allowed, (payload["kind"], key)
                assert len(value) not in {300, 250, 350}


def _per_site_statistics(transport) -> set[float]:
    values: set[float] = set()
    for record in transport.log.records:
        message = json.loads(record.raw)
        payload = message.get("payload") or {}
        kind = payload.get("kind")
        if kind == "feature_relevance":
            values.update(payload["score"] + payload["information"])
            values.update(
                u / i ** 0.5 for u, i in zip(payload["score"], payload["information"],
                                            strict=True) if i > 0
            )
        elif kind == "feature_covariance":
            values.update(payload["scatter_upper"])
    return {v for v in values if v not in (0.0, 1.0)}


def test_released_result_contains_no_per_site_statistics(run):
    _, outcome, transport = run
    per_site = _per_site_statistics(transport)
    released = set(_numbers(json.loads(outcome.model_dump_json())))
    assert per_site and not (per_site & released)
    # Leave-one-site-out releases selections, never scores.
    loso = _strategy(outcome, "sequential_stratified").leave_one_site_out
    assert all(isinstance(v, tuple) for v in loso.selections.values())


def test_clients_receive_no_statistics_of_other_sites(run):
    _, _, transport = run
    per_site = _per_site_statistics(transport)
    for record in transport.log.records:
        if record.direction is Direction.TO_CLIENT:
            assert not (per_site & set(_numbers(json.loads(record.raw))))


def test_payload_contracts():
    with pytest.raises(ValidationError, match="p\\(p\\+1\\)/2"):
        FeatureCovariancePayload(feature_names=("a", "b"), n_samples=20,
                                 rank_transformed=False, scatter_upper=(1.0, 0.5))
    with pytest.raises(ValidationError, match="diagonal"):
        FeatureCovariancePayload(feature_names=("a", "b"), n_samples=20,
                                 rank_transformed=False, scatter_upper=(1.0, 0.5, -1.0))
    with pytest.raises(ValidationError, match="\\[-1, 1\\]"):
        FeatureReliabilityPayload(feature_names=("a",), column="min_icc", n_subjects=20,
                                  values=(1.5,))
    with pytest.raises(ValidationError, match="exceed"):
        FeatureRelevancePayload(feature_names=("a",), n_samples=20, n_events=30,
                                score=(1.0,), information=(2.0,))
    with pytest.raises(ValidationError, match="one entry per feature"):
        FeatureRelevancePayload(feature_names=("a",), n_samples=20, n_events=10,
                                score=tuple(range(20)), information=(2.0,))
    ok = FeatureReliabilityPayload(feature_names=("a", "b"), column="min_icc", n_subjects=20,
                                   values=(0.9, None))
    assert ok.values[1] is None


class _Tamper:
    def __init__(self, inner: FederatedClient, tamper) -> None:
        self.client_id, self._inner, self._tamper = inner.client_id, inner, tamper

    def build_registration(self):
        return self._inner.build_registration()

    def handle_registration_ack(self, raw):
        self._inner.handle_registration_ack(raw)

    def handle_round(self, raw):
        message = json.loads(self._inner.handle_round(raw))
        self._tamper(message)
        return json.dumps(message).encode()


def _few_events(message):
    if message["task"] == "feature_relevance":
        message["payload"]["n_events"] = 1


def _wrong_transform(message):
    if message["task"] == "feature_covariance":
        message["payload"]["rank_transformed"] = False


def test_coordinator_rejects_non_conforming_selection_updates():
    config = stage4()
    clients = build_clients("exp-s4", config)
    for client in clients:
        client.initialize()
    transport = InProcessTransport()
    for client in [_Tamper(clients[0], _few_events), *clients[1:]]:
        transport.attach(client)
    from coordinator.engine import FederatedCoordinator

    outcome = FederatedCoordinator("exp-s4", config, transport).run()
    relevance = next(r for r in outcome.rounds if r.task is TaskKind.FEATURE_RELEVANCE)
    assert relevance.clients[0].reason_code.value == "disclosure_blocked"
    assert relevance.aggregate.n_clients == 2

    transport = InProcessTransport()
    for client in [_Tamper(clients[0], _wrong_transform), *clients[1:]]:
        transport.attach(client)
    outcome = FederatedCoordinator("exp-s4", config, transport).run()
    assert "different correlation transform" in outcome.rounds[-1].failure_reason
    assert not outcome.completed
