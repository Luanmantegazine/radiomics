"""API response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from coordinator.harmonization.protocol import HarmonizationResult
from coordinator.selection.results import FederatedSelectionResult
from shared.protocol import MetricsSummary, SiteMetrics


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ExperimentOut(ApiModel):
    id: str
    name: str
    status: str
    data_origin: str
    protocol_version: str
    seed: int
    config_hash: str
    mlflow_run_id: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentDetail(ExperimentOut):
    config: dict[str, Any]


class ClientOut(ApiModel):
    experiment_id: str
    client_id: str
    status: str
    reason_code: str | None
    data_origin: str
    software: dict[str, Any]
    registered_at: datetime


class ParticipationOut(ApiModel):
    client_id: str
    status: str
    reason_code: str | None
    n_samples: int | None


class RoundOut(ApiModel):
    id: str
    experiment_id: str
    round_number: int
    phase: str
    task: str
    aggregator: str
    status: str
    failure_reason: str | None
    finished_at: datetime


class RoundDetail(RoundOut):
    participations: list[ParticipationOut]


class RoundResult(ApiModel):
    round_id: str
    round_number: int
    phase: str
    task: str
    aggregator: str
    status: str
    aggregate: dict[str, Any] | None


class ResultsOut(ApiModel):
    experiment_id: str
    status: str
    data_origin: str
    """'synthetic' results validate infrastructure only; they are not science."""
    config_hash: str
    rounds: list[RoundResult]


class CostOut(ApiModel):
    round_id: str | None
    scope: str
    client_id: str | None
    metric: str
    value: float
    unit: str


class SelectedFeatureOut(ApiModel):
    strategy: str
    rank: int
    feature: str
    ranking_score: float
    reliability: float | None
    relevance_z: float | None
    relevance_p: float | None
    agreement: float | None


class ModelOut(ApiModel):
    id: int
    experiment_id: str
    name: str
    kind: str
    status: str
    failure_reason: str | None
    features_source: str
    feature_names: list[str]
    mlflow_model_uri: str | None
    created_at: datetime


class ModelDetail(ModelOut):
    parameters: dict[str, Any] | None


# --------------------------------------------------------------------------- #
# Stage 7: consolidated, typed responses
# --------------------------------------------------------------------------- #
class ValidationIssue(BaseModel):
    location: str
    message: str


class ValidationOut(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = []
    config_hash: str | None = None
    phases: list[str] = []
    expected_clients: list[str] = []
    data_origin: str | None = None


class TemplateOut(BaseModel):
    id: str
    name: str
    description: str
    data_origin: str
    phases: list[str]
    config: dict[str, Any]


class RoundCounts(BaseModel):
    total: int
    failed: int
    by_phase: dict[str, int]


class ExperimentSummary(BaseModel):
    experiment: ExperimentOut
    registered_clients: int
    rejected_clients: int
    rounds: RoundCounts
    planned_phases: list[str]
    available_results: dict[str, bool]
    """Which result sections exist: harmonization, feature_selection, models, evaluation."""
    total_bytes: float | None
    duration_s: float | None


class PhaseCost(BaseModel):
    phase: str
    rounds: int
    bytes_from_clients: float
    bytes_to_clients: float
    communication_s: float
    aggregation_s: float
    round_time_s: float


class ClientCost(BaseModel):
    client_id: str
    rounds: int
    bytes_sent: float
    bytes_received: float
    local_execution_s: float
    peak_memory_bytes: float | None


class CostSummary(BaseModel):
    experiment_id: str
    totals: dict[str, float]
    """Experiment-level measurements (total time and bytes, time per phase...)."""
    by_phase: list[PhaseCost]
    by_client: list[ClientCost]
    by_round: list[dict[str, Any]]
    """Round number, phase, task, bytes in each direction and duration."""


class FeatureSelectionOut(FederatedSelectionResult):
    experiment_id: str
    data_origin: str


class HarmonizationOut(HarmonizationResult):
    experiment_id: str
    data_origin: str


class EvaluationRow(BaseModel):
    model_name: str
    variant: str
    scope: str
    n_clients: int
    feature_names: list[str]
    pooled: list[MetricsSummary]
    sites: list[SiteMetrics]


class EvaluationOut(BaseModel):
    experiment_id: str
    data_origin: str
    evaluations: list[EvaluationRow]
