"""API response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


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
