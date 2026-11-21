"""Coordinator engine: registration and round orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from coordinator.aggregation import AggregationError, get_strategy
from coordinator.evaluation.protocol import EvaluationResult, FederatedEvaluationProtocol
from coordinator.harmonization.protocol import (
    FederatedHarmonizationProtocol,
    HarmonizationResult,
)
from coordinator.ingress import IngressRejected, validate_update
from coordinator.outcomes import ClientRoundOutcome, ExperimentOutcome, RoundOutcome
from coordinator.ports import NullRecorder, RunRecorder
from coordinator.registry import ClientRegistry
from coordinator.selection.protocol import FederatedSelectionProtocol, RoundRequest
from coordinator.selection.results import FederatedSelectionResult
from coordinator.training.protocol import FederatedTrainingProtocol, TrainingResult
from coordinator.transport import Reply
from shared.config import ExperimentConfig, config_hash
from shared.logging import get_logger
from shared.protocol import (
    ClientRegistration,
    ClientUpdate,
    EvaluationRequest,
    FeatureInventoryAggregate,
    FederatedPhase,
    HarmonizationState,
    ModelState,
    PartitionSpec,
    ReasonCode,
    RegistrationAck,
    RoundConfig,
    TaskKind,
    UpdateStatus,
    WireModel,
    decode,
    encode,
)
from shared.seeding import derive_seed

Aggregate = Callable[[Sequence[ClientUpdate]], WireModel]


class Transport(Protocol):
    """Subset of the transport the engine depends on."""

    @property
    def client_ids(self) -> tuple[str, ...]: ...

    def fetch_registration(self, client_id: str) -> bytes: ...

    def send_registration_ack(self, client_id: str, raw: bytes) -> None: ...

    def send_round(self, client_id: str, raw: bytes) -> bytes: ...

    def exchange(self, client_ids: tuple[str, ...], raw: bytes) -> dict[str, Reply]:
        """Send one round to several sites and collect their answers."""
        ...


class RoundProtocol(Protocol):
    """A multi-round protocol (feature selection, training) driven by the engine."""

    def next_request(self) -> RoundRequest | None: ...

    def aggregate(self, request: RoundRequest, updates: Sequence[ClientUpdate]) -> WireModel: ...

    def on_round_failed(self, request: RoundRequest, reason: str) -> bool:
        """Return True to continue with the next request."""
        ...


class FederatedCoordinator:
    """Runs one experiment: admits clients, then executes the configured rounds.

    State carried between fixed rounds is limited to *global aggregates* (the
    agreed feature schema). During feature selection the protocol object keeps
    per-site statistics in memory until the selection result is computed; they
    are never persisted or returned.
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
        selection: FederatedSelectionResult | None = None
        training: TrainingResult | None = None
        harmonization: HarmonizationResult | None = None
        evaluation: EvaluationResult | None = None
        selection_s = training_s = harmonization_s = evaluation_s = 0.0
        completed = len(registered) >= self._config.privacy.min_clients_per_aggregate
        if not completed:
            self._log.error("not_enough_clients", registered=len(registered))
        else:
            for spec in self._config.rounds:
                outcome = self._run_round(
                    len(rounds) + 1, spec.phase, spec.task, dict(spec.params),
                    spec.aggregator, self._registry_aggregate(spec.aggregator, spec.task),
                    registered,
                )
                rounds.append(outcome)
                self._recorder.round_finished(outcome)
                if not outcome.completed:
                    completed = False
                    break
            if completed and self._config.harmonization is not None:
                started_h = time.perf_counter()
                completed, harmonization = self._run_harmonization(rounds, registered)
                harmonization_s = time.perf_counter() - started_h
            if completed and self._config.feature_selection is not None:
                completed, selection, selection_s = self._run_selection(rounds, registered)
            if completed and self._config.training is not None:
                completed, training, training_s = self._run_training(
                    rounds, registered, selection
                )
            if completed and self._config.evaluation is not None and training is not None:
                started_e = time.perf_counter()
                evaluation = self._run_evaluation(rounds, registered, training)
                evaluation_s = time.perf_counter() - started_e
        result = ExperimentOutcome(
            experiment_id=self._experiment_id,
            completed=completed,
            registered_clients=registered,
            rounds=tuple(rounds),
            feature_selection=selection,
            selection_s=selection_s,
            training=training,
            training_s=training_s,
            harmonization=harmonization,
            harmonization_s=harmonization_s,
            evaluation=evaluation,
            evaluation_s=evaluation_s,
            total_bytes=sum(c.bytes_sent + c.bytes_received for r in rounds for c in r.clients),
            duration_s=time.perf_counter() - started,
        )
        self._recorder.experiment_finished(result)
        return result

    def _run_selection(
        self, rounds: list[RoundOutcome], clients: tuple[str, ...]
    ) -> tuple[bool, FederatedSelectionResult | None, float]:
        protocol = FederatedSelectionProtocol(
            self._config.feature_selection, self._config.privacy, self._feature_names
        )
        if not self._drive(protocol, rounds, clients):
            return False, None, 0.0
        started = time.perf_counter()
        try:
            selection = protocol.result()
        except AggregationError as exc:
            self._log.error("selection_failed", reason=str(exc))
            return False, None, time.perf_counter() - started
        duration = time.perf_counter() - started
        self._log.info(
            "selection_finished",
            strategies=len(selection.strategies),
            duration_s=round(duration, 4),
        )
        self._recorder.selection_finished(self._experiment_id, selection)
        return True, selection, duration

    def _run_harmonization(
        self, rounds: list[RoundOutcome], clients: tuple[str, ...]
    ) -> tuple[bool, HarmonizationResult | None]:
        protocol = FederatedHarmonizationProtocol(
            self._config.harmonization, self._config.privacy, self._feature_names
        )
        try:
            if not self._drive(protocol, rounds, clients):
                return False, None
        except AggregationError as exc:
            self._log.error("harmonization_failed", reason=str(exc))
            return False, None
        self._feature_names = protocol.schema
        result = protocol.result()
        self._recorder.harmonization_finished(self._experiment_id, result)
        return True, result

    def _run_evaluation(
        self, rounds: list[RoundOutcome], clients: tuple[str, ...], training: TrainingResult
    ) -> EvaluationResult:
        config = self._config.evaluation
        protocol = FederatedEvaluationProtocol(config, self._config.training, training.models)
        self._drive(protocol, rounds, clients)
        evaluations = list(protocol.result().evaluations)
        if config.external_validation == "leave_one_site_out":
            evaluations += self._external_validation(rounds, clients)
        result = EvaluationResult(evaluations=tuple(evaluations))
        self._recorder.evaluation_finished(self._experiment_id, result)
        return result

    def _external_validation(self, rounds: list[RoundOutcome], clients: tuple[str, ...]):
        """Re-run selection and training without each site, evaluate on that site.

        Harmonization is not re-estimated per fold (it uses no outcome; see docs).
        """
        out = []
        for held_out in clients:
            subset = tuple(c for c in clients if c != held_out)
            if len(subset) < self._config.privacy.min_clients_per_aggregate:
                self._log.warning("external_validation_skipped", held_out=held_out)
                continue
            selection = None
            if self._config.feature_selection is not None:
                selection_protocol = FederatedSelectionProtocol(
                    self._config.feature_selection, self._config.privacy, self._feature_names
                )
                if not self._drive(selection_protocol, rounds, subset):
                    continue
                try:
                    selection = selection_protocol.result()
                except AggregationError:
                    continue
            training_protocol = FederatedTrainingProtocol(self._config.training, selection)
            self._drive(training_protocol, rounds, subset)
            protocol = FederatedEvaluationProtocol(
                self._config.evaluation, self._config.training,
                training_protocol.result().models, scope=f"external:{held_out}",
                clients=(held_out,), variants=("released",), evaluation_partition="all",
            )
            self._drive(protocol, rounds, subset)
            out.extend(protocol.result().evaluations)
        return out

    def _run_training(
        self,
        rounds: list[RoundOutcome],
        clients: tuple[str, ...],
        selection: FederatedSelectionResult | None,
    ) -> tuple[bool, TrainingResult | None, float]:
        started = time.perf_counter()
        protocol = FederatedTrainingProtocol(self._config.training, selection)
        if not self._drive(protocol, rounds, clients):  # pragma: no cover - training continues
            return False, None, 0.0
        try:
            training = protocol.result()
        except AggregationError as exc:
            self._log.error("training_failed", reason=str(exc))
            return False, None, time.perf_counter() - started
        duration = time.perf_counter() - started
        self._log.info(
            "training_finished",
            trained=sum(m.status == "trained" for m in training.models),
            failed=sum(m.status == "failed" for m in training.models),
        )
        self._recorder.training_finished(self._experiment_id, training)
        return True, training, duration

    def _partition(self) -> PartitionSpec | None:
        split = self._config.data_split
        if split is None:
            return None
        return PartitionSpec(name="train", test_fraction=split.test_fraction, seed=split.seed)

    def _drive(
        self, protocol: RoundProtocol, rounds: list[RoundOutcome], clients: tuple[str, ...]
    ) -> bool:
        """Run the protocol's rounds; False when a failed round stops it."""
        while (request := protocol.next_request()) is not None:
            outcome = self._run_round(
                len(rounds) + 1, request.phase, request.task, request.params,
                request.aggregator, self._protocol_aggregate(protocol, request), clients,
                feature_names=request.feature_names,
                partition=self._partition(),
                model=request.model,
                harmonization=request.harmonization,
                evaluation=request.evaluation,
                min_clients=request.min_clients,
                round_clients=request.clients,
            )
            rounds.append(outcome)
            self._recorder.round_finished(outcome)
            if not outcome.completed and not protocol.on_round_failed(
                request, outcome.failure_reason or "failed"
            ):
                return False
        return True

    @staticmethod
    def _protocol_aggregate(protocol: RoundProtocol, request: RoundRequest):
        return lambda updates: protocol.aggregate(request, updates)

    @staticmethod
    def _registry_aggregate(name: str, task: TaskKind) -> Aggregate:
        def aggregate(updates: Sequence[ClientUpdate]) -> WireModel:
            strategy = get_strategy(name)
            if strategy.task is not task:
                raise AggregationError(f"strategy {strategy.name} does not aggregate {task.value}")
            return strategy.aggregate(updates)

        return aggregate

    def _run_round(
        self,
        number: int,
        phase: FederatedPhase,
        task: TaskKind,
        params: dict,
        aggregator: str,
        aggregate_fn: Aggregate,
        clients: tuple[str, ...],
        *,
        feature_names: tuple[str, ...] | None = None,
        partition: PartitionSpec | None = None,
        model: ModelState | None = None,
        harmonization: HarmonizationState | None = None,
        evaluation: EvaluationRequest | None = None,
        min_clients: int | None = None,
        round_clients: tuple[str, ...] | None = None,
    ) -> RoundOutcome:
        if round_clients is not None:
            clients = round_clients
        started = time.perf_counter()
        config = RoundConfig(
            protocol_version=self._config.protocol_version,
            experiment_id=self._experiment_id,
            round_id=f"{self._experiment_id}-r{number:03d}",
            round_number=number,
            phase=phase,
            task=task,
            seed=derive_seed(self._config.seed, "round", number),
            config_hash=self._config_hash,
            privacy=self._config.privacy,
            feature_names=feature_names if feature_names is not None else self._feature_names,
            params=params,
            partition=partition,
            model=model,
            harmonization=harmonization,
            evaluation=evaluation,
        )
        raw_config = encode(config)
        log = self._log.bind(round_id=config.round_id, task=task.value)
        log.info("round_started", n_clients=len(clients))

        accepted: list[ClientUpdate] = []
        client_outcomes: list[ClientRoundOutcome] = []
        communication = 0.0
        replies = self._transport.exchange(tuple(clients), raw_config)
        for client_id in clients:
            reply = replies[client_id]
            raw_update, round_trip = reply.raw, reply.round_trip_s
            if raw_update is None:
                log.warning("no_response", client_id=client_id)
                client_outcomes.append(
                    ClientRoundOutcome(
                        client_id=client_id,
                        status=UpdateStatus.FAILED,
                        reason_code=ReasonCode.NO_RESPONSE,
                        bytes_received=len(raw_config),
                        round_trip_s=round_trip,
                    )
                )
                continue
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
        required = min_clients or self._config.privacy.min_clients_per_aggregate
        if len(accepted) < required:
            failure = "too_few_valid_updates"
        else:
            agg_started = time.perf_counter()
            try:
                aggregate = aggregate_fn(accepted)
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
            phase=phase,
            task=task,
            aggregator=aggregator,
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
