"""Isolation tests: individual-level data never crosses the federated boundary.

Every byte exchanged between the simulated clients and the coordinator is
captured and inspected against the clients' actual local data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pytest

from client.node import FederatedClient
from coordinator.engine import FederatedCoordinator
from coordinator.ingress import IngressRejected, validate_update
from coordinator.transport import Direction, InProcessTransport
from experiments.simulation import build_clients
from shared.protocol import (
    PROTOCOL_VERSION,
    ClientRegistration,
    ClientUpdate,
    PrivacyPolicy,
    RoundConfig,
    UpdateStatus,
    decode,
    encode,
)
from tests.federated.conftest import with_updates

EXPERIMENT_ID = "exp-isolation"


def _run(config, clients=None):
    transport = InProcessTransport(capture=True)
    clients = clients if clients is not None else build_clients(EXPERIMENT_ID, config)
    for client in clients:
        if isinstance(client, FederatedClient):
            client.initialize()
        transport.attach(client)
    outcome = FederatedCoordinator(EXPERIMENT_ID, config, transport).run()
    return outcome, transport, clients


def _numbers(node) -> Iterator[float]:
    if isinstance(node, bool):
        return
    if isinstance(node, int | float):
        yield float(node)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _numbers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _numbers(value)


def _arrays(node) -> Iterator[list]:
    if isinstance(node, list):
        yield node
        for value in node:
            yield from _arrays(value)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _arrays(value)


@pytest.fixture
def run(config):
    return _run(config)


def test_flow_completes_and_traffic_is_captured(run):
    outcome, transport, _ = run
    assert outcome.completed
    assert all(record.raw is not None for record in transport.log.records)
    assert transport.log.total_bytes(Direction.TO_COORDINATOR) > 0


def test_no_participant_identifier_leaves_a_client(run):
    _, transport, clients = run
    wire = b"\n".join(r.raw for r in transport.log.records).decode()
    for client in clients:
        for participant_id in client._dataset.participant_ids:
            assert participant_id not in wire
    assert "SYN-" not in wire


def test_no_individual_value_leaves_a_client(run):
    """No feature value or follow-up time of any participant appears on the wire."""
    _, transport, clients = run
    individual = np.concatenate(
        [np.concatenate([c._dataset.features.ravel(), c._dataset.time_months]) for c in clients]
    )
    sent = np.array(
        [
            number
            for record in transport.log.records
            if record.direction is Direction.TO_COORDINATOR
            for number in _numbers(json.loads(record.raw))
        ]
    )
    assert sent.size > 0
    closest = np.abs(sent[:, None] - individual[None, :]).min()
    assert closest > 1e-9


def test_no_message_carries_a_per_participant_array(run):
    _, transport, clients = run
    cohort_sizes = {c._dataset.n_participants for c in clients}
    max_features = max(len(c._dataset.feature_names) for c in clients)
    for record in transport.log.records:
        for array in _arrays(json.loads(record.raw)):
            assert len(array) not in cohort_sizes
            if any(isinstance(v, int | float) and not isinstance(v, bool) for v in array):
                assert len(array) <= max_features  # numeric arrays are feature-indexed


def test_messages_contain_only_allowlisted_fields(run):
    _, transport, _ = run
    forbidden = {"participant_ids", "time_months", "event", "features", "labels", "image", "mask"}
    for record in transport.log.records:
        message = json.loads(record.raw)
        assert forbidden.isdisjoint(message)
        assert forbidden.isdisjoint(message.get("payload") or {})


def test_clients_never_receive_other_clients_statistics(run):
    """What a client receives contains no per-site statistic of another site."""
    _, transport, _ = run
    per_site_numbers: dict[str, set[float]] = {}
    for record in transport.log.records:
        if record.direction is Direction.TO_COORDINATOR:
            message = json.loads(record.raw)
            if message.get("message_type") == "client_update":
                per_site_numbers.setdefault(record.client_id, set()).update(
                    _numbers(message.get("payload") or {})
                )
    for record in transport.log.records:
        if record.direction is Direction.TO_CLIENT:
            message = json.loads(record.raw)
            received = set(_numbers({k: v for k, v in message.items() if k != "privacy"}))
            received -= {float(message.get("round_number", -1)), float(message.get("seed", -1))}
            for other, numbers in per_site_numbers.items():
                if other != record.client_id:
                    assert not {n for n in received & numbers if not n.is_integer()}
            assert other_client_ids(record.client_id, transport).isdisjoint(_strings(message))


def other_client_ids(client_id, transport) -> set[str]:
    return set(transport.client_ids) - {client_id}


def _strings(node) -> set[str]:
    if isinstance(node, str):
        return {node}
    if isinstance(node, dict):
        return set().union(*(_strings(v) for v in node.values()), set())
    if isinstance(node, list):
        return set().union(*(_strings(v) for v in node), set())
    return set()


def test_transport_has_no_client_to_client_route():
    public = {name for name in dir(InProcessTransport) if not name.startswith("_")}
    assert public == {
        "attach", "client_ids", "fetch_registration", "send_registration_ack", "send_round",
    }
    assert {d.value for d in Direction} == {"client_to_coordinator", "coordinator_to_client"}


def test_small_cohort_refuses_to_report_statistics(config):
    """A site below the minimum cohort size sends a refusal code and no numbers."""
    tight = with_updates(config, privacy={"min_cohort_size": 50, "min_clients_per_aggregate": 2})
    outcome, transport, _ = _run(tight)
    summary_round = outcome.rounds[1]
    by_client = {c.client_id: c for c in summary_round.clients}
    assert by_client["client-02"].status is UpdateStatus.REFUSED  # 45 participants < 50
    assert by_client["client-02"].reason_code.value == "cohort_too_small"
    assert summary_round.completed and summary_round.aggregate.n_clients == 2
    refusal = [
        json.loads(r.raw)
        for r in transport.log.records
        if r.client_id == "client-02" and r.direction is Direction.TO_COORDINATOR
    ][-1]
    assert refusal["payload"] is None


def test_aggregate_of_a_single_site_is_never_released(config):
    """With one valid site the 'aggregate' would equal that site's statistics."""
    tight = with_updates(config, privacy={"min_cohort_size": 70, "min_clients_per_aggregate": 2})
    outcome, _, _ = _run(tight)  # only client-03 (80) is large enough
    summary_round = outcome.rounds[1]
    assert not summary_round.completed
    assert summary_round.aggregate is None
    assert summary_round.failure_reason == "too_few_valid_updates"
    assert not outcome.completed


class _LeakyClient:
    """A misbehaving endpoint that tries to push individual data upstream."""

    def __init__(self, inner: FederatedClient, tamper) -> None:
        self.client_id = inner.client_id
        self._inner, self._tamper = inner, tamper

    def build_registration(self) -> bytes:
        return self._inner.build_registration()

    def handle_registration_ack(self, raw: bytes) -> None:
        self._inner.handle_registration_ack(raw)

    def handle_round(self, raw: bytes) -> bytes:
        message = json.loads(self._inner.handle_round(raw))
        self._tamper(message)
        return json.dumps(message).encode()


def _add_participant_rows(message: dict) -> None:
    message["participants"] = [{"id": "P1", "time_months": 12.0, "event": True}]


def _per_participant_means(message: dict) -> None:
    if message["task"] == "feature_summary":
        message["payload"]["mean"] = [0.5] * 60


def _tiny_cohort(message: dict) -> None:
    if message["task"] == "feature_summary":
        message["payload"]["n_samples"] = 3


@pytest.mark.parametrize("tamper", [_add_participant_rows, _per_participant_means, _tiny_cohort])
def test_coordinator_discards_non_conforming_updates(config, tamper):
    """Defence in depth: ingress rejects what a faulty client lets through."""
    clients = build_clients(EXPERIMENT_ID, config)
    for client in clients:
        client.initialize()
    clients[0] = _LeakyClient(clients[0], tamper)
    outcome, _, _ = _run(config, clients)
    rejected = outcome.rounds[-1].clients[0]
    assert rejected.status is UpdateStatus.REFUSED
    assert rejected.reason_code.value == "disclosure_blocked"
    assert rejected.n_samples is None
    aggregate = outcome.rounds[-1].aggregate
    assert aggregate is None or aggregate.n_clients == 2


def test_ingress_rejects_updates_from_the_wrong_sender():
    config = RoundConfig(
        protocol_version=PROTOCOL_VERSION, experiment_id="e", round_id="e-r001", round_number=1,
        phase="infrastructure", task="feature_inventory", seed=1, config_hash="h",
        privacy=PrivacyPolicy(),
    )
    update = ClientUpdate(
        protocol_version=PROTOCOL_VERSION, experiment_id="e", round_id="e-r001",
        client_id="client-02", task="feature_inventory", status="completed",
        payload={"kind": "feature_inventory", "feature_names": ["a"]},
    )
    with pytest.raises(IngressRejected, match="sender mismatch"):
        validate_update(encode(update), sender="client-01", config=config)


def test_egress_guard_blocks_local_identifiers(config):
    """Even if a task tried to emit a participant id, the guard stops it."""
    from client.egress import EgressGuard
    from shared.privacy import DisclosureViolation

    guard = EgressGuard(["OAS30001"])
    update = ClientUpdate(
        protocol_version=PROTOCOL_VERSION, experiment_id="e", round_id="e-r001",
        client_id="client-01", task="feature_inventory", status="completed",
        payload={"kind": "feature_inventory", "feature_names": ["OAS30001"]},
    )
    with pytest.raises(DisclosureViolation):
        guard.authorize(update, PrivacyPolicy())


def test_registration_discloses_no_cohort_information(run):
    _, transport, _ = run
    first = next(r for r in transport.log.records if r.direction is Direction.TO_COORDINATOR)
    registration = decode(first.raw, ClientRegistration)
    assert set(registration.model_dump()) == {
        "message_type", "protocol_version", "experiment_id", "client_id", "supported_tasks",
        "data_origin", "software",
    }
