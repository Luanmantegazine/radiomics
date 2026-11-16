"""Outbound port of the coordinator: where run progress is recorded.

The backend implements this (PostgreSQL + MLflow); the coordinator itself has
no persistence dependency.
"""

from __future__ import annotations

from typing import Protocol

from coordinator.outcomes import ExperimentOutcome, RoundOutcome
from shared.protocol import ClientRegistration, ReasonCode


class RunRecorder(Protocol):
    def experiment_started(self, experiment_id: str) -> None: ...

    def client_registered(
        self, registration: ClientRegistration, accepted: bool, reason: ReasonCode | None
    ) -> None: ...

    def round_finished(self, outcome: RoundOutcome) -> None: ...

    def experiment_finished(self, outcome: ExperimentOutcome) -> None: ...


class NullRecorder:
    """Recorder that discards everything (tests, dry runs)."""

    def experiment_started(self, experiment_id: str) -> None:
        return None

    def client_registered(
        self, registration: ClientRegistration, accepted: bool, reason: ReasonCode | None
    ) -> None:
        return None

    def round_finished(self, outcome: RoundOutcome) -> None:
        return None

    def experiment_finished(self, outcome: ExperimentOutcome) -> None:
        return None
