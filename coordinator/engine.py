"""Coordinator engine: registration and round orchestration."""

from __future__ import annotations

import time
from typing import Protocol

from coordinator.aggregation import AggregationError, get_strategy
from coordinator.ingress import IngressRejected, validate_update
from coordinator.outcomes import ClientRoundOutcome, ExperimentOutcome, RoundOutcome
from coordinator.ports import NullRecorder, RunRecorder
from coordinator.registry import ClientRegistry
from shared.config import ExperimentConfig, RoundSpec, config_hash
from shared.logging import get_logger
from shared.protocol import (
    ClientRegistration,
    ClientUpdate,
    FeatureInventoryAggregate,
    ReasonCode,
    RegistrationAck,
    RoundConfig,
    UpdateStatus,
    decode,
    encode,
)
from shared.seeding import derive_seed


class Transport(Protocol):
    """Subset of the transport the engine depends on."""

    @property
    def client_ids(self) -> tuple[str, ...]: ...

    def fetch_registration(self, client_id: str) -> bytes: ...

    def send_registration_ack(self, client_id: str, raw: bytes) -> None: ...

    def send_round(self, client_id: str, raw: bytes) -> bytes: ...


class FederatedCoordinator:
    """Runs one experiment: admits clients, then executes the configured rounds.

    State carried between rounds is limited to *global aggregates* (currently
    the agreed feature schema); individual client updates are discarded after
    aggregation.
    """

    def __init__(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        transport: Transport,
        recorder: RunRecorder | None = None,
    ) -> None:
        self._experiment_id = experiment_id
        self._config = config
        self._config_hash = config_hash(config)
        self._transport = transport
        self._recorder: RunRecorder = recorder or NullRecorder()
        self._registry = ClientRegistry(experiment_id, config)
        self._feature_names: tuple[str, ...] | None = None
        self._log = get_logger("coordinator").bind(experiment_id=experiment_id)

    # -- registration ------------------------------------------------------- #
    def register_clients(self) -> tuple[str, ...]:
        for client_id in self._transport.client_ids:
            try:
                registration = decode(
                    self._transport.fetch_registration(client_id), ClientRegistration
                )
            except ValueError:
                self._log.warning("registration_rejected", client_id=client_id, reason="schema")
                continue
            reason = (
                ReasonCode.UNKNOWN_CLIENT
                if registration.client_id != client_id
                else self._registry.admit(registration)
            )
            ack = RegistrationAck(
                protocol_version=self._config.protocol_version,
                experiment_id=self._experiment_id,
                client_id=client_id,
                accepted=reason is None,
                reason_code=reason,
            )
            self._transport.send_registration_ack(client_id, encode(ack))
            self._recorder.client_registered(registration, reason is None, reason)
            self._log.info(
                "client_registration",
                client_id=client_id,
                accepted=reason is None,
                reason=reason.value if reason else None,
            )
        return self._registry.client_ids

    # -- rounds ------------------------------------------------------------- #
    def run(self) -> ExperimentOutcome:
        started = time.perf_counter()
        self._recorder.experiment_started(self._experiment_id)
        registered = self.register_clients()
        rounds: list[RoundOutcome] = []
        completed = len(registered) >= self._config.privacy.min_clients_per_aggregate
        if not completed:
            self._log.error("not_enough_clients", registered=len(registered))
        else:
            for number, spec in enumerate(self._config.rounds, start=1):
                outcome = self._run_round(number, spec, registered)
                rounds.append(outcome)
                self._recorder.round_finished(outcome)
                if not outcome.completed:
                    completed = False
                    break
        result = ExperimentOutcome(
            experiment_id=self._experiment_id,
            completed=completed,
            registered_clients=registered,
            rounds=tuple(rounds),
            total_bytes=sum(c.bytes_sent + c.bytes_received for r in rounds for c in r.clients),
            duration_s=time.perf_counter() - started,
        )
        self._recorder.experiment_finished(result)
        return result

    def _round_config(self, number: int, spec: RoundSpec) -> RoundConfig:
        return RoundConfig(
            protocol_version=self._config.protocol_version,
            experiment_id=self._experiment_id,
            round_id=f"{self._experiment_id}-r{number:03d}",
            round_number=number,
            phase=spec.phase,
            task=spec.task,
            seed=derive_seed(self._config.seed, "round", number),
            config_hash=self._config_hash,
            privacy=self._config.privacy,
            feature_names=self._feature_names,
            params=dict(spec.params),
        )

    def _run_round(self, number: int, spec: RoundSpec, clients: tuple[str, ...]) -> RoundOutcome:
        started = time.perf_counter()
        config = self._round_config(number, spec)
        raw_config = encode(config)
        log = self._log.bind(round_id=config.round_id, task=spec.task.value)
        log.info("round_started", n_clients=len(clients))

        accepted: list[ClientUpdate] = []
        client_outcomes: list[ClientRoundOutcome] = []
        communication = 0.0
        for client_id in clients:
            sent = time.perf_counter()
            raw_update = self._transport.send_round(client_id, raw_config)
            round_trip = time.perf_counter() - sent
            try:
                update = validate_update(raw_update, sender=client_id, config=config)
            except IngressRejected as exc:
                log.warning("update_rejected", client_id=client_id, reason=str(exc))
                client_outcomes.append(
                    ClientRoundOutcome(
                        client_id=client_id,
                        status=UpdateStatus.REFUSED,
                        reason_code=ReasonCode.DISCLOSURE_BLOCKED,
                        bytes_sent=len(raw_update),
                        bytes_received=len(raw_config),
                        round_trip_s=round_trip,
                    )
                )
                continue
            local = sum(update.costs.durations_s.values())
            communication += max(round_trip - local, 0.0)
            if update.status is UpdateStatus.COMPLETED:
                accepted.append(update)
            client_outcomes.append(
                ClientRoundOutcome(
                    client_id=client_id,
                    status=update.status,
                    reason_code=update.reason_code,
                    n_samples=getattr(update.payload, "n_samples", None),
                    bytes_sent=len(raw_update),
                    bytes_received=len(raw_config),
                    round_trip_s=round_trip,
                    costs=update.costs,
                )
            )

        aggregate, failure, aggregation_s = None, None, 0.0
        if len(accepted) < self._config.privacy.min_clients_per_aggregate:
            failure = "too_few_valid_updates"
        else:
            agg_started = time.perf_counter()
            try:
                strategy = get_strategy(spec.aggregator)
                if strategy.task is not spec.task:
                    raise AggregationError(
                        f"strategy {strategy.name} does not aggregate {spec.task.value}"
                    )
                aggregate = strategy.aggregate(accepted)
            except AggregationError as exc:
                failure = f"aggregation_error: {exc}"
            aggregation_s = time.perf_counter() - agg_started
        if isinstance(aggregate, FeatureInventoryAggregate):
            self._feature_names = aggregate.feature_names
        del accepted  # individual updates are not retained beyond aggregation

        outcome = RoundOutcome(
            experiment_id=self._experiment_id,
            round_id=config.round_id,
            round_number=number,
            phase=spec.phase,
            task=spec.task,
            aggregator=spec.aggregator,
            completed=failure is None,
            failure_reason=failure,
            aggregate=aggregate,
            clients=tuple(client_outcomes),
            communication_s=communication,
            aggregation_s=aggregation_s,
            duration_s=time.perf_counter() - started,
        )
        log.info("round_finished", completed=outcome.completed, failure=failure)
        return outcome
