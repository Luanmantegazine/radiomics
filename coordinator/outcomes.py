"""Result types produced by the coordinator and consumed by recorders."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from coordinator.selection.results import FederatedSelectionResult
from coordinator.training.protocol import TrainingResult
from shared.protocol import (
    CostReport,
    FederatedPhase,
    GlobalAggregate,
    ReasonCode,
    TaskKind,
    UpdateStatus,
)


class _Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClientRoundOutcome(_Outcome):
    """What is retained about one client in one round: metadata and costs only."""

    client_id: str
    status: UpdateStatus
    reason_code: ReasonCode | None = None
    n_samples: int | None = None
    bytes_sent: int = 0
    """Bytes the client sent to the coordinator."""
    bytes_received: int = 0
    round_trip_s: float = 0.0
    costs: CostReport = Field(default_factory=CostReport)


class RoundOutcome(_Outcome):
    experiment_id: str
    round_id: str
    round_number: int
    phase: FederatedPhase
    task: TaskKind
    aggregator: str
    completed: bool
    failure_reason: str | None = None
    aggregate: GlobalAggregate | None = None
    clients: tuple[ClientRoundOutcome, ...] = ()
    communication_s: float = 0.0
    """Round-trip time minus client-reported local execution (serialization and
    transport overhead; close to zero in the in-process simulation)."""
    aggregation_s: float = 0.0
    duration_s: float = 0.0


class ExperimentOutcome(_Outcome):
    experiment_id: str
    completed: bool
    registered_clients: tuple[str, ...]
    rounds: tuple[RoundOutcome, ...]
    feature_selection: FederatedSelectionResult | None = None
    selection_s: float = 0.0
    """Coordinator time spent evaluating the selection strategies (after the rounds)."""
    training: TrainingResult | None = None
    training_s: float = 0.0
    """Wall-clock time of the whole training phase, rounds included."""
    total_bytes: int
    duration_s: float
