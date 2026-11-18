"""Federated client node: registration and round execution.

The node exposes a *bytes-in / bytes-out* interface so that the exact same code
runs behind any transport (in-process simulation today; HTTP/gRPC/Flower later).
"""

from __future__ import annotations

from importlib import metadata

from client.data import LocalDataset, LocalDataSource
from client.egress import EgressGuard
from client.splits import partition
from client.tasks import LocalTask, TaskRefused, default_tasks
from shared.costs import CostTracker
from shared.logging import get_logger
from shared.privacy import DisclosureViolation
from shared.protocol import (
    PROTOCOL_VERSION,
    ClientRegistration,
    ClientUpdate,
    ReasonCode,
    RegistrationAck,
    RoundConfig,
    TaskKind,
    UpdateStatus,
    decode,
    encode,
)


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pydantic"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:  # pragma: no cover
            continue
    return versions


class FederatedClient:
    """One participating site."""

    def __init__(
        self,
        client_id: str,
        experiment_id: str,
        data_source: LocalDataSource,
        tasks: dict[TaskKind, LocalTask] | None = None,
    ) -> None:
        self.client_id = client_id
        self._experiment_id = experiment_id
        self._data_source = data_source
        self._tasks = tasks if tasks is not None else default_tasks()
        self._dataset: LocalDataset | None = None
        self._guard: EgressGuard | None = None
        self._registered = False
        self._log = get_logger("client").bind(client_id=client_id, experiment_id=experiment_id)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self) -> None:
        """Load local data. Nothing is transmitted."""
        self._dataset = self._data_source.load()
        self._guard = EgressGuard(self._dataset.participant_ids)
        self._log.info("client_initialized", n_features=len(self._dataset.feature_names))

    def build_registration(self) -> bytes:
        return encode(
            ClientRegistration(
                protocol_version=PROTOCOL_VERSION,
                experiment_id=self._experiment_id,
                client_id=self.client_id,
                supported_tasks=tuple(self._tasks),
                data_origin=self._data_source.origin,
                software=_software_versions(),
            )
        )

    def handle_registration_ack(self, raw: bytes) -> None:
        ack = decode(raw, RegistrationAck)
        self._registered = ack.accepted and ack.client_id == self.client_id
        self._log.info("registration_ack", accepted=self._registered)

    # -- rounds ------------------------------------------------------------- #
    def handle_round(self, raw: bytes) -> bytes:
        """Run one round and return the serialized, authorized update."""
        config = decode(raw, RoundConfig)
        costs = CostTracker()
        status, reason, payload = UpdateStatus.COMPLETED, None, None
        try:
            with costs.stage("local_execution"):
                payload = self._execute(config)
        except (TaskRefused, DisclosureViolation) as exc:
            status, reason = UpdateStatus.REFUSED, exc.code
        except Exception:  # noqa: BLE001 - never let details of local data escape
            self._log.exception("local_task_failed", round_id=config.round_id)
            status, reason = UpdateStatus.FAILED, ReasonCode.INTERNAL_ERROR

        def build(status: UpdateStatus, reason: ReasonCode | None, payload) -> ClientUpdate:
            return ClientUpdate(
                protocol_version=PROTOCOL_VERSION,
                experiment_id=config.experiment_id,
                round_id=config.round_id,
                client_id=self.client_id,
                task=config.task,
                status=status,
                reason_code=reason,
                payload=payload,
                costs=costs.report(),
            )

        try:
            if self._guard is None:
                raise DisclosureViolation(ReasonCode.NOT_REGISTERED, "client not initialized")
            out = self._guard.authorize(build(status, reason, payload), config.privacy)
        except (DisclosureViolation, ValueError) as exc:
            code = getattr(exc, "code", ReasonCode.DISCLOSURE_BLOCKED)
            status = UpdateStatus.REFUSED
            self._log.warning("egress_blocked", round_id=config.round_id, reason=code.value)
            out = encode(build(status, code, None))
        self._log.info(
            "round_executed", round_id=config.round_id, task=config.task.value, status=status.value
        )
        return out

    def _execute(self, config: RoundConfig):
        if config.protocol_version != PROTOCOL_VERSION:
            raise TaskRefused(ReasonCode.PROTOCOL_MISMATCH)
        if config.experiment_id != self._experiment_id:
            raise TaskRefused(ReasonCode.EXPERIMENT_MISMATCH)
        if not self._registered or self._dataset is None:
            raise TaskRefused(ReasonCode.NOT_REGISTERED)
        task = self._tasks.get(config.task)
        if task is None:
            raise TaskRefused(ReasonCode.UNSUPPORTED_TASK)
        return task.run(partition(self._dataset, config.partition), config)
