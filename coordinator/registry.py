"""Registry of participating clients for one experiment."""

from __future__ import annotations

from shared.config import ExperimentConfig
from shared.protocol import ClientRegistration, ReasonCode, TaskKind


def selection_tasks(config: ExperimentConfig) -> set[TaskKind]:
    """Tasks a site must support to take part in the configured selection."""
    selection = config.feature_selection
    if selection is None:
        return set()
    tasks = {TaskKind.FEATURE_INVENTORY, TaskKind.FEATURE_RELEVANCE}
    if selection.reliability_column is not None:
        tasks.add(TaskKind.FEATURE_RELIABILITY)
    if selection.redundancy is not None:
        tasks.add(TaskKind.FEATURE_COVARIANCE)
    return tasks


def training_tasks(config: ExperimentConfig) -> set[TaskKind]:
    if config.training is None:
        return set()
    tasks = {TaskKind.FEATURE_SUMMARY, TaskKind.MODEL_UPDATE}
    if any(entry.model.kind == "cox" for entry in config.training.models):
        tasks.add(TaskKind.BASELINE_HAZARD)
    if any(
        getattr(entry.model, "intercept", None) == "site_specific"
        for entry in config.training.models
    ):
        tasks.add(TaskKind.SITE_INTERCEPT)
    return tasks


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
        required = (
            {spec.task for spec in self._config.rounds}
            | selection_tasks(self._config)
            | training_tasks(self._config)
        )
        if not required.issubset(registration.supported_tasks):
            return ReasonCode.UNSUPPORTED_TASK
        self._accepted[registration.client_id] = registration
        return None

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(self._accepted)
