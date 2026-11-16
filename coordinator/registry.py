"""Registry of participating clients for one experiment."""

from __future__ import annotations

from shared.config import ExperimentConfig
from shared.protocol import ClientRegistration, ReasonCode


class ClientRegistry:
    """Admission control: only allowlisted, protocol-compatible sites join."""

    def __init__(self, experiment_id: str, config: ExperimentConfig) -> None:
        self._experiment_id = experiment_id
        self._config = config
        self._accepted: dict[str, ClientRegistration] = {}

    def admit(self, registration: ClientRegistration) -> ReasonCode | None:
        """Return ``None`` when accepted, otherwise the refusal code."""
        if registration.protocol_version != self._config.protocol_version:
            return ReasonCode.PROTOCOL_MISMATCH
        if registration.experiment_id != self._experiment_id:
            return ReasonCode.EXPERIMENT_MISMATCH
        if registration.client_id not in self._config.expected_clients:
            return ReasonCode.UNKNOWN_CLIENT
        if registration.client_id in self._accepted:
            return ReasonCode.DUPLICATE_CLIENT
        if registration.data_origin is not self._config.data_origin:
            # Never mix synthetic and real sites in one experiment.
            return ReasonCode.DATA_ORIGIN_MISMATCH
        required = {spec.task for spec in self._config.rounds}
        if not required.issubset(registration.supported_tasks):
            return ReasonCode.UNSUPPORTED_TASK
        self._accepted[registration.client_id] = registration
        return None

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(self._accepted)
