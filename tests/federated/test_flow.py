"""End-to-end flow of the simulated federation (no backend)."""

from __future__ import annotations

import numpy as np

from coordinator.transport import InProcessTransport
from experiments.simulation import build_clients, run_simulation
from tests.federated.conftest import with_updates


class RecordingRecorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def experiment_started(self, experiment_id):
        self.events.append(("started", experiment_id))

    def client_registered(self, registration, accepted, reason):
        self.events.append(("registered", registration.client_id, accepted))

    def round_finished(self, outcome):
        self.events.append(("round", outcome.round_number, outcome.completed))

    def experiment_finished(self, outcome):
        self.events.append(("finished", outcome.completed))


def test_full_flow(config):
    recorder = RecordingRecorder()
    outcome = run_simulation("exp-flow", config, recorder=recorder)

    assert outcome.completed
    assert outcome.registered_clients == ("client-01", "client-02", "client-03")
    assert recorder.events == [
        ("started", "exp-flow"),
        ("registered", "client-01", True),
        ("registered", "client-02", True),
        ("registered", "client-03", True),
        ("round", 1, True),
        ("round", 2, True),
        ("finished", True),
    ]
    inventory, summary = (r.aggregate for r in outcome.rounds)
    # feat_f is missing at client-03, so the agreed schema drops it ...
    assert inventory.feature_names == ("feat_a", "feat_b", "feat_c", "feat_d", "feat_e")
    # ... and round 2 was configured with the schema agreed in round 1.
    assert summary.feature_names == inventory.feature_names
    assert summary.n_samples == 60 + 45 + 80
    assert outcome.total_bytes > 0
    assert all(c.costs.durations_s for r in outcome.rounds for c in r.clients)


def test_federated_aggregate_matches_centralized_computation(config):
    """Infrastructure correctness check on synthetic data (not a scientific result)."""
    outcome = run_simulation("exp-flow", config)
    summary = outcome.rounds[1].aggregate
    datasets = [c._data_source.load() for c in build_clients("exp-flow", config)]
    columns = [
        d.features[:, [d.feature_names.index(n) for n in summary.feature_names]] for d in datasets
    ]
    pooled = np.concatenate(columns)
    np.testing.assert_allclose(summary.mean, pooled.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(summary.variance, pooled.var(axis=0, ddof=1), rtol=1e-10)


def test_runs_are_reproducible_and_seed_dependent(config):
    first = run_simulation("exp-a", config).rounds[1].aggregate
    second = run_simulation("exp-a", config).rounds[1].aggregate
    other = run_simulation("exp-a", with_updates(config, seed=config.seed + 1)).rounds[1].aggregate
    assert first == second
    assert first.mean != other.mean


def test_unlisted_client_is_not_admitted(config):
    restricted = with_updates(config, expected_clients=["client-01", "client-02"])
    recorder = RecordingRecorder()
    outcome = run_simulation("exp-flow", restricted, recorder=recorder)
    assert outcome.registered_clients == ("client-01", "client-02")
    assert ("registered", "client-03", False) in recorder.events
    assert all(c.client_id != "client-03" for r in outcome.rounds for c in r.clients)
    assert outcome.rounds[1].aggregate.n_samples == 60 + 45


def test_run_stops_without_enough_clients(config):
    transport = InProcessTransport()
    lonely = with_updates(
        config, simulation={"clients": [config.simulation.clients[0].model_dump(mode="json")]}
    )
    outcome = run_simulation("exp-flow", lonely, transport=transport)
    assert not outcome.completed
    assert outcome.rounds == ()
