"""Versioned experiment configuration (YAML -> validated Pydantic models)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.protocol import (
    PROTOCOL_VERSION,
    ClientId,
    DataOrigin,
    FeatureName,
    FederatedPhase,
    PrivacyPolicy,
    TaskKind,
    Token,
)

CONFIG_SCHEMA_VERSION = 1


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoundSpec(ConfigModel):
    """One federated round: which local task runs and how results are aggregated."""

    phase: FederatedPhase
    task: TaskKind
    aggregator: Token
    params: dict[Token, bool | int | float | str | None] = Field(default_factory=dict)


class SyntheticDataSourceSpec(ConfigModel):
    """Synthetic local dataset. For infrastructure tests only, never for science."""

    kind: Literal["synthetic"] = "synthetic"
    n_participants: int = Field(ge=1)
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)


# Future stages add real sources (e.g. a NIfTI directory) to this union.
DataSourceSpec = Annotated[SyntheticDataSourceSpec, Field(discriminator="kind")]


class SimulatedClientSpec(ConfigModel):
    client_id: ClientId
    data_source: DataSourceSpec


class SimulationSpec(ConfigModel):
    """Single-machine simulation settings.

    In a distributed deployment this section does not exist: each institution
    keeps its own local data configuration, which the coordinator never sees.
    """

    clients: tuple[SimulatedClientSpec, ...] = Field(min_length=1)


class ExperimentConfig(ConfigModel):
    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    name: Token
    description: str = Field(default="", max_length=2000)
    seed: int = Field(ge=0)
    protocol_version: Token = PROTOCOL_VERSION
    data_origin: DataOrigin
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    expected_clients: tuple[ClientId, ...] = Field(min_length=2)
    """Allowlist of pseudonymised sites permitted to register."""
    rounds: tuple[RoundSpec, ...] = Field(min_length=1)
    simulation: SimulationSpec | None = None

    @model_validator(mode="after")
    def _consistency(self) -> ExperimentConfig:
        if len(set(self.expected_clients)) != len(self.expected_clients):
            raise ValueError("expected_clients must be unique")
        if len(self.expected_clients) < self.privacy.min_clients_per_aggregate:
            raise ValueError("fewer expected clients than privacy.min_clients_per_aggregate")
        if self.simulation is not None:
            simulated = [c.client_id for c in self.simulation.clients]
            if len(set(simulated)) != len(simulated):
                raise ValueError("simulated client ids must be unique")
            if self.data_origin is not DataOrigin.SYNTHETIC:
                raise ValueError("a simulated experiment must declare data_origin: synthetic")
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""
    with Path(path).open(encoding="utf-8") as handle:
        return ExperimentConfig.model_validate(yaml.safe_load(handle))


def config_hash(config: ExperimentConfig) -> str:
    """Stable short hash of a configuration, used to version every result."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
